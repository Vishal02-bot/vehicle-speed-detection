"""Secure, one-time terminal setup for the project Admin account."""

from __future__ import annotations

import argparse
import getpass
import sys

from src.auth_store import (
    AdminAccountAlreadyExists,
    auth_database_path,
    authenticate_admin,
    create_admin_account,
    get_admin_account_status,
    has_admin_account,
    reset_admin_password,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="reset an existing Admin password from this local terminal",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="show the configured database path and non-secret Admin status",
    )
    args = parser.parse_args()

    if args.diagnose:
        print(f"Admin database: {auth_database_path()}")
        username = input("Admin username: ")
        account = get_admin_account_status(username)
        if account is None:
            print("No Admin account exists for that username in this database.")
            return 1
        print(f"Account: {account.username}")
        print(f"Role: {account.role}")
        print(f"Status: {account.status}")
        print(f"Failed sign-in attempts: {account.failed_attempts}")
        print(f"Locked: {'yes' if account.is_locked else 'no'}")
        return 0

    if args.reset_password:
        print(f"Admin database: {auth_database_path()}")
        username = input("Admin username: ")
        password = getpass.getpass("New Admin password (minimum 12 characters): ")
        confirmation = getpass.getpass("Confirm new Admin password: ")
        if password != confirmation:
            print("Passwords do not match.")
            return 1
        try:
            updated = reset_admin_password(username, password)
        except ValueError as exc:
            print(f"Admin password reset failed: {exc}")
            return 1
        if not updated:
            print("No matching active Admin account exists; no changes were made.")
            return 1
        account = authenticate_admin(username, password)
        if account is None or account.role != "Admin":
            print(
                "The password hash was updated, but Admin sign-in verification failed. "
                "Use --diagnose to check account status and database path."
            )
            return 1
        print("Admin password reset successfully.")
        return 0

    if has_admin_account():
        print(f"Admin database: {auth_database_path()}")
        print("An Admin account already exists. First-time setup is disabled.")
        print(
            "Use --diagnose to inspect its status or --reset-password "
            "to securely choose a new password."
        )
        return 1

    print(f"Admin database: {auth_database_path()}")
    print("Create the single Admin account. Password input is hidden.")
    username = input("Admin username: ")
    password = getpass.getpass("Admin password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm Admin password: ")
    if password != confirmation:
        print("Passwords do not match.")
        return 1

    try:
        account = create_admin_account(username, password)
    except (ValueError, AdminAccountAlreadyExists) as exc:
        print(f"Admin setup failed: {exc}")
        return 1

    print(f"Admin account '{account.username}' created successfully.")
    print("Start the app with: python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
