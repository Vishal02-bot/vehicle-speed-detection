"""Securely reset the existing Admin username and password."""

from __future__ import annotations

import getpass
import sqlite3
import sys

from src.auth_store import (
    auth_database_path,
    backup_auth_database,
    get_admin_account_status,
    reset_admin_credentials,
    validate_admin_credentials,
)


def main() -> int:
    print(f"Admin database: {auth_database_path()}")
    current_username = input("Current Admin username: ").strip()
    account = get_admin_account_status(current_username)
    if account is None:
        print("No Admin account exists for that username; no changes were made.")
        return 1
    if account.role != "Admin":
        print("The existing account does not have the Admin role; no changes were made.")
        return 1
    if account.status != "active":
        print("The existing Admin account is not active; no changes were made.")
        return 1

    new_username = input("New Admin username: ")
    new_password = getpass.getpass("New Admin password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm new Admin password: ")
    if new_password != confirmation:
        print("Passwords do not match; no changes were made.")
        return 1

    try:
        normalized_username, validated_password = validate_admin_credentials(
            new_username, new_password
        )
        backup_path = backup_auth_database()
        updated_account = reset_admin_credentials(
            current_username,
            normalized_username,
            validated_password,
        )
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"Admin credential reset failed: {exc}")
        return 1

    print(f"Database backup: {backup_path}")
    print(f"Admin credentials updated for '{updated_account.username}'.")
    print("Account status: active")
    print("Account role: Admin")
    print("New Admin credentials verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
