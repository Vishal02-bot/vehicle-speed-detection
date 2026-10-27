"""Tests for the secure existing-Admin credential reset command."""

import sqlite3

import pytest

import src.admin_reset as admin_reset
from src.auth_store import (
    PASSWORD_HASHER,
    authenticate_admin,
    backup_auth_database,
    create_admin_account,
    get_admin_account_status,
)


@pytest.fixture
def reset_database(tmp_path, monkeypatch):
    database_path = tmp_path / "auth" / "admin.sqlite3"
    monkeypatch.setenv("VEHICLE_AUTH_DB", str(database_path))
    return database_path


def _insert_admin(database_path, username="old-admin"):
    create_admin_account(username, "original-admin-password")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE admin_accounts SET created_at = '2024-01-02 03:04:05'"
        )


def test_reset_utility_updates_existing_account_and_preserves_metadata(
    reset_database, monkeypatch, capsys
):
    _insert_admin(reset_database)
    answers = iter(("old-admin", "new_admin.1"))
    passwords = iter(("replacement-secure-password", "replacement-secure-password"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        admin_reset.getpass,
        "getpass",
        lambda _prompt: next(passwords),
    )

    assert admin_reset.main() == 0
    output = capsys.readouterr().out
    assert "Admin credentials updated for 'new_admin.1'." in output
    assert "Account status: active" in output
    assert "Account role: Admin" in output
    assert "replacement-secure-password" not in output

    with sqlite3.connect(reset_database) as connection:
        rows = connection.execute(
            """
            SELECT username, role, status, created_at, password_hash
            FROM admin_accounts
            """
        ).fetchall()
    assert len(rows) == 1
    username, role, status, created_at, password_hash = rows[0]
    assert (username, role, status, created_at) == (
        "new_admin.1",
        "Admin",
        "active",
        "2024-01-02 03:04:05",
    )
    assert PASSWORD_HASHER.verify(password_hash, "replacement-secure-password")
    assert authenticate_admin("old-admin", "original-admin-password") is None
    assert authenticate_admin("new_admin.1", "replacement-secure-password").role == "Admin"

    backups = list(reset_database.parent.joinpath("backups").glob("admin-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(f"{backups[0].as_uri()}?mode=ro", uri=True) as backup:
        backup_admin = backup.execute(
            "SELECT username, role, status FROM admin_accounts"
        ).fetchone()
    assert backup_admin == ("old-admin", "Admin", "active")


@pytest.mark.parametrize(
    ("new_username", "new_password"),
    [
        ("ab", "replacement-secure-password"),
        ("invalid username", "replacement-secure-password"),
        ("valid-admin", "short"),
    ],
)
def test_reset_rejects_invalid_new_credentials_without_changing_database(
    reset_database, monkeypatch, capsys, new_username, new_password
):
    _insert_admin(reset_database)
    answers = iter(("old-admin", new_username))
    passwords = iter((new_password, new_password))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        admin_reset.getpass,
        "getpass",
        lambda _prompt: next(passwords),
    )

    assert admin_reset.main() == 1
    assert "Admin credential reset failed" in capsys.readouterr().out
    assert authenticate_admin("old-admin", "original-admin-password").role == "Admin"
    assert not reset_database.parent.joinpath("backups").exists()


def test_reset_refuses_unknown_or_disabled_account_without_database_changes(
    reset_database, monkeypatch, capsys
):
    _insert_admin(reset_database)
    answers = iter(("unknown-admin",))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert admin_reset.main() == 1
    assert "No Admin account exists" in capsys.readouterr().out
    assert authenticate_admin("old-admin", "original-admin-password").role == "Admin"
    assert not reset_database.parent.joinpath("backups").exists()

    with sqlite3.connect(reset_database) as connection:
        connection.execute(
            "UPDATE admin_accounts SET status = 'disabled' WHERE username = 'old-admin'"
        )
    answers = iter(("old-admin",))
    assert admin_reset.main() == 1
    assert "not active" in capsys.readouterr().out
    assert get_admin_account_status("old-admin").status == "disabled"


def test_sqlite_backup_is_unique_and_keeps_original_account(reset_database):
    _insert_admin(reset_database)

    first_backup = backup_auth_database()
    second_backup = backup_auth_database()

    assert first_backup != second_backup
    with sqlite3.connect(f"{first_backup.as_uri()}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute(
            "SELECT username, role, status FROM admin_accounts"
        ).fetchone() == ("old-admin", "Admin", "active")
