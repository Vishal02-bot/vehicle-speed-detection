"""SQLite-backed, single-role Admin authentication."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

PASSWORD_HASHER = PasswordHasher(type=Type.ID)
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{3,64}$")


@dataclass(frozen=True)
class AdminAccount:
    username: str
    role: str


@dataclass(frozen=True)
class AdminAccountStatus:
    username: str
    role: str
    status: str
    failed_attempts: int
    is_locked: bool


class AdminAccountAlreadyExists(Exception):
    """Raised when first-time setup is attempted after an Admin exists."""


def _database_path() -> Path:
    configured_path = os.environ.get("VEHICLE_AUTH_DB")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data" / "auth" / "admin.sqlite3"


def auth_database_path() -> Path:
    """Return the resolved SQLite path shared by setup, login, and the API."""
    return _database_path()


def backup_auth_database() -> Path:
    """Create a unique, verified SQLite backup without exposing account hashes."""
    source_path = _database_path()
    if not source_path.is_file():
        raise FileNotFoundError(f"Admin database not found: {source_path}")

    backup_dir = source_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        backup_dir.chmod(0o700)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"admin-{timestamp}-{uuid4().hex}.sqlite3"
    descriptor = os.open(
        backup_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    os.close(descriptor)

    try:
        with sqlite3.connect(f"{source_path.as_uri()}?mode=ro", uri=True) as source:
            with sqlite3.connect(backup_path) as destination:
                source.backup(destination)
        with sqlite3.connect(f"{backup_path.as_uri()}?mode=ro", uri=True) as check:
            integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise sqlite3.DatabaseError(
                    f"SQLite backup integrity check failed: {integrity}"
                )
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise

    return backup_path


def _connect() -> sqlite3.Connection:
    database_path = _database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_accounts (
            username TEXT PRIMARY KEY COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role = 'Admin'),
            status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    if os.name != "nt":
        database_path.chmod(0o600)
    return connection


@contextmanager
def _database() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def validate_admin_credentials(username: str, password: str) -> tuple[str, str]:
    """Validate first-time Admin setup inputs without modifying the database."""
    normalized_username = username.strip()
    if not USERNAME_PATTERN.fullmatch(normalized_username):
        raise ValueError(
            "Username must be 3–64 characters using letters, numbers, '.', '_' or '-'."
        )
    if len(password) < 12:
        raise ValueError("Admin password must contain at least 12 characters.")
    if len(password) > 1024:
        raise ValueError("Admin password is too long.")
    return normalized_username, password


def has_admin_account() -> bool:
    with _database() as connection:
        return (
            connection.execute(
                "SELECT 1 FROM admin_accounts WHERE role = 'Admin' LIMIT 1"
            ).fetchone()
            is not None
        )


def get_admin_account_status(username: str) -> AdminAccountStatus | None:
    """Return non-secret status for the matching Admin account."""
    normalized_username = username.strip()
    with _database() as connection:
        row = connection.execute(
            """
            SELECT username, role, status, failed_attempts, locked_until
            FROM admin_accounts
            WHERE username = ? AND role = 'Admin'
            """,
            (normalized_username,),
        ).fetchone()
    if row is None:
        return None
    return AdminAccountStatus(
        username=row["username"],
        role=row["role"],
        status=row["status"],
        failed_attempts=row["failed_attempts"],
        is_locked=row["locked_until"] is not None and row["locked_until"] > time.time(),
    )


def create_admin_account(username: str, password: str) -> AdminAccount:
    """Create the one-time Admin account; passwords are stored only as Argon2id."""
    normalized_username, validated_password = validate_admin_credentials(
        username, password
    )
    password_hash = PASSWORD_HASHER.hash(validated_password)

    with _database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM admin_accounts WHERE role = 'Admin' LIMIT 1"
        ).fetchone():
            raise AdminAccountAlreadyExists(
                "An Admin account already exists; first-time setup is disabled."
            )
        connection.execute(
            """
            INSERT INTO admin_accounts
                (username, password_hash, role, status, created_at)
            VALUES (?, ?, 'Admin', 'active', datetime('now'))
            """,
            (normalized_username, password_hash),
        )
    return AdminAccount(username=normalized_username, role="Admin")


def reset_admin_password(username: str, password: str) -> bool:
    """Reset an active Admin password from the trusted local terminal."""
    normalized_username, validated_password = validate_admin_credentials(
        username, password
    )
    password_hash = PASSWORD_HASHER.hash(validated_password)
    with _database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        account = connection.execute(
            """
            SELECT 1 FROM admin_accounts
            WHERE username = ? AND role = 'Admin' AND status = 'active'
            """,
            (normalized_username,),
        ).fetchone()
        if account is None:
            return False
        cursor = connection.execute(
            """
            UPDATE admin_accounts
            SET password_hash = ?, failed_attempts = 0, locked_until = NULL
            WHERE username = ? AND role = 'Admin' AND status = 'active'
            """,
            (password_hash, normalized_username),
        )
        return cursor.rowcount == 1


def reset_admin_credentials(
    current_username: str,
    new_username: str,
    new_password: str,
) -> AdminAccount:
    """Update an active Admin row in place, preserving its creation metadata."""
    normalized_current_username = current_username.strip()
    normalized_new_username, validated_password = validate_admin_credentials(
        new_username, new_password
    )
    password_hash = PASSWORD_HASHER.hash(validated_password)

    with _database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        account = connection.execute(
            """
            SELECT role, status FROM admin_accounts
            WHERE username = ?
            """,
            (normalized_current_username,),
        ).fetchone()
        if account is None:
            raise ValueError("No account exists for the supplied current username.")
        if account["role"] != "Admin":
            raise ValueError("The existing account does not have the Admin role.")
        if account["status"] != "active":
            raise ValueError("The existing Admin account is not active.")

        try:
            cursor = connection.execute(
                """
                UPDATE admin_accounts
                SET username = ?, password_hash = ?,
                    failed_attempts = 0, locked_until = NULL
                WHERE username = ? AND role = 'Admin' AND status = 'active'
                """,
                (
                    normalized_new_username,
                    password_hash,
                    normalized_current_username,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "The new username is already assigned to an account."
            ) from exc

        if cursor.rowcount != 1:
            raise RuntimeError("The Admin account was not updated.")

    verified_account = authenticate_admin(normalized_new_username, validated_password)
    if verified_account is None or verified_account.role != "Admin":
        raise RuntimeError(
            "The updated account could not be verified as an active Admin."
        )
    return verified_account


def authenticate_admin(username: str, password: str) -> AdminAccount | None:
    """Authenticate an active Admin, applying a per-account login lockout."""
    normalized_username = username.strip()
    with _database() as connection:
        account = connection.execute(
            """
            SELECT username, password_hash, role, status, failed_attempts, locked_until
            FROM admin_accounts
            WHERE username = ?
            """,
            (normalized_username,),
        ).fetchone()
        if account is None:
            return None

        try:
            password_matches = PASSWORD_HASHER.verify(
                account["password_hash"], password
            )
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            password_matches = False

        now = time.time()
        if (
            account["status"] != "active"
            or account["role"] != "Admin"
            or (account["locked_until"] is not None and account["locked_until"] > now)
            or not password_matches
        ):
            failed_attempts = account["failed_attempts"] + 1
            locked_until = (
                now + LOCKOUT_SECONDS
                if failed_attempts >= MAX_LOGIN_ATTEMPTS
                else account["locked_until"]
            )
            connection.execute(
                """
                UPDATE admin_accounts
                SET failed_attempts = ?, locked_until = ?
                WHERE username = ?
                """,
                (failed_attempts, locked_until, account["username"]),
            )
            return None

        connection.execute(
            """
            UPDATE admin_accounts
            SET failed_attempts = 0, locked_until = NULL
            WHERE username = ?
            """,
            (account["username"],),
        )
        return AdminAccount(username=account["username"], role=account["role"])


def is_active_admin(username: str, role: str) -> bool:
    """Re-check the stored role and account status before allowing access."""
    with _database() as connection:
        return (
            connection.execute(
                """
                SELECT 1 FROM admin_accounts
                WHERE username = ? AND role = 'Admin' AND status = 'active'
                """,
                (username.strip(),),
            ).fetchone()
            is not None
            and role == "Admin"
        )
