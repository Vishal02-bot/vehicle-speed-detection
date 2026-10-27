"""End-to-end tests for the Admin-only Streamlit access gate."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.auth_store import create_admin_account

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_unauthenticated_dashboard_login_and_logout(tmp_path, monkeypatch):
    monkeypatch.setenv("VEHICLE_AUTH_DB", str(tmp_path / "admin.sqlite3"))
    create_admin_account("traffic-admin", "very-secure-password")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["admin_username"] = "traffic-admin"
    app.session_state["admin_role"] = "User"
    app.run()

    assert not app.exception
    assert any(element.value == "Admin Login" for element in app.subheader)
    assert not app.radio

    app.text_input(key="admin_login_username").set_value("traffic-admin")
    app.text_input(key="admin_login_password").set_value("wrong-password")
    app.button(key="admin_login_submit").click().run()

    assert any(
        "Sign-in failed" in element.value for element in app.error
    )
    assert not app.radio

    app.text_input(key="admin_login_username").set_value("traffic-admin")
    app.text_input(key="admin_login_password").set_value("very-secure-password")
    app.button(key="admin_login_submit").click().run()

    assert not app.exception
    assert app.session_state["admin_role"] == "Admin"
    assert app.session_state.get("admin_login_password", "") != "very-secure-password"
    assert len(app.radio) == 1

    app.button(key="admin_logout").click().run()

    assert not app.exception
    assert any(element.value == "Admin Login" for element in app.subheader)
    assert not app.radio


def test_missing_admin_shows_only_local_setup_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("VEHICLE_AUTH_DB", str(tmp_path / "admin.sqlite3"))

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert any(element.value == "Admin Login" for element in app.subheader)
    assert any(
        "python -m src.admin_setup" in element.value for element in app.warning
    )
    assert not app.text_input
    assert not app.radio
