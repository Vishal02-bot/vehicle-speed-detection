"""Tests for Admin account setup, password hashing, and authorization."""

import sqlite3

import pytest

import src.admin_setup as admin_setup
from src.auth_store import (
    AdminAccountAlreadyExists,
    authenticate_admin,
    create_admin_account,
    get_admin_account_status,
    has_admin_account,
    is_active_admin,
    reset_admin_password,
)


@pytest.fixture
def auth_database(tmp_path, monkeypatch):
    database_path = tmp_path / "private" / "admin.sqlite3"
    monkeypatch.setenv("VEHICLE_AUTH_DB", str(database_path))
    return database_path


def test_first_time_setup_stores_argon2id_hash_and_only_admin_role(auth_database):
    assert not has_admin_account()

    account = create_admin_account("traffic-admin", "very-secure-password")

    assert account.username == "traffic-admin"
    assert account.role == "Admin"
    assert authenticate_admin("traffic-admin", "very-secure-password") == account
    assert authenticate_admin("traffic-admin", "wrong-password") is None

    with sqlite3.connect(auth_database) as connection:
        stored_hash, role, status = connection.execute(
            "SELECT password_hash, role, status FROM admin_accounts"
        ).fetchone()
    assert stored_hash.startswith("$argon2id$")
    assert stored_hash != "very-secure-password"
    assert role == "Admin"
    assert status == "active"


def test_first_time_setup_cannot_create_a_second_or_user_account(auth_database):
    create_admin_account("traffic-admin", "very-secure-password")

    with pytest.raises(AdminAccountAlreadyExists):
        create_admin_account("another-user", "another-secure-password")

    with sqlite3.connect(auth_database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM admin_accounts"
        ).fetchone()[0] == 1


def test_terminal_setup_prompts_without_echoing_or_printing_password(
    auth_database, monkeypatch, capsys
):
    answers = iter(("traffic-admin",))
    passwords = iter(("very-secure-password", "very-secure-password"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(admin_setup.getpass, "getpass", lambda _prompt: next(passwords))
    monkeypatch.setattr("sys.argv", ["admin_setup"])

    assert admin_setup.main() == 0
    output = capsys.readouterr().out
    assert "traffic-admin" in output
    assert "very-secure-password" not in output
    assert authenticate_admin("traffic-admin", "very-secure-password") is not None


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("ab", "very-secure-password"),
        ("bad username", "very-secure-password"),
        ("valid-admin", "short"),
    ],
)
def test_setup_rejects_invalid_username_or_password(auth_database, username, password):
    with pytest.raises(ValueError):
        create_admin_account(username, password)

    assert not has_admin_account()


def test_disabled_account_and_non_admin_role_never_grant_access(auth_database):
    account = create_admin_account("traffic-admin", "very-secure-password")
    with sqlite3.connect(auth_database) as connection:
        connection.execute(
            "UPDATE admin_accounts SET status = 'disabled' WHERE username = ?",
            (account.username,),
        )

    assert authenticate_admin("traffic-admin", "very-secure-password") is None
    assert not is_active_admin("traffic-admin", "Admin")
    assert not is_active_admin("traffic-admin", "User")


def test_local_password_reset_changes_only_the_existing_admin_password(auth_database):
    create_admin_account("traffic-admin", "very-secure-password")

    assert reset_admin_password("traffic-admin", "new-secure-password")
    assert authenticate_admin("traffic-admin", "very-secure-password") is None
    assert authenticate_admin("traffic-admin", "new-secure-password") is not None
    assert not reset_admin_password("unknown-admin", "another-secure-password")


def test_terminal_reset_verifies_new_password_without_echoing_it(
    auth_database, monkeypatch, capsys
):
    create_admin_account("traffic-admin", "very-secure-password")
    answers = iter((" traffic-admin ",))
    passwords = iter(("new-secure-password", "new-secure-password"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(admin_setup.getpass, "getpass", lambda _prompt: next(passwords))
    monkeypatch.setattr("sys.argv", ["admin_setup", "--reset-password"])

    assert admin_setup.main() == 0
    output = capsys.readouterr().out
    assert "Admin password reset successfully" in output
    assert "new-secure-password" not in output
    assert authenticate_admin("traffic-admin", "very-secure-password") is None
    assert authenticate_admin("traffic-admin", "new-secure-password") is not None


def test_reset_rejects_disabled_admin_without_changing_existing_password(auth_database):
    account = create_admin_account("traffic-admin", "very-secure-password")
    with sqlite3.connect(auth_database) as connection:
        connection.execute(
            "UPDATE admin_accounts SET status = 'disabled' WHERE username = ?",
            (account.username,),
        )

    assert get_admin_account_status(" TRAFFIC-ADMIN ").status == "disabled"
    assert not reset_admin_password("traffic-admin", "new-secure-password")
    assert authenticate_admin("traffic-admin", "very-secure-password") is None


def test_repeated_invalid_logins_apply_account_lockout(auth_database):
    create_admin_account("traffic-admin", "very-secure-password")

    for _ in range(5):
        assert authenticate_admin("traffic-admin", "incorrect-password") is None

    assert authenticate_admin("traffic-admin", "very-secure-password") is None
