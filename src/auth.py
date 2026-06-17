"""Lightweight local authentication: username/password accounts + sessions.

Self-contained and free - no external service. Passwords are hashed with
PBKDF2-HMAC-SHA256 (Python stdlib), and users/sessions are persisted in a small
SQLite database so accounts survive restarts. Everywhere else in the app, each
account's papers, chats and stats are isolated by `username`.
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager

import config

_PBKDF2_ROUNDS = 200_000
_USERNAME_RE = re.compile(r"^[a-z0-9_]{3,24}$")
_SESSION_TTL = 60 * 60 * 24 * 30  # 30 days


class AuthError(Exception):
    """Invalid credentials or a registration problem (shown to the user)."""


@contextmanager
def _db():
    conn = sqlite3.connect(str(config.AUTH_DB))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init():
    with _db() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "username TEXT PRIMARY KEY, salt BLOB NOT NULL, "
            "pw_hash BLOB NOT NULL, created REAL NOT NULL)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "token TEXT PRIMARY KEY, username TEXT NOT NULL, created REAL NOT NULL)"
        )


_init()


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def register(username: str, password: str) -> str:
    """Create a new account and return the normalized username."""
    username = normalize_username(username)
    if not _USERNAME_RE.match(username):
        raise AuthError("Username must be 3–24 characters: lowercase letters, numbers or underscore.")
    if len(password or "") < 6:
        raise AuthError("Password must be at least 6 characters.")
    salt = os.urandom(16)
    pw_hash = _hash(password, salt)
    try:
        with _db() as c:
            c.execute(
                "INSERT INTO users (username, salt, pw_hash, created) VALUES (?,?,?,?)",
                (username, salt, pw_hash, time.time()),
            )
    except sqlite3.IntegrityError:
        raise AuthError("That username is already taken.")
    return username


def verify(username: str, password: str) -> str:
    """Check credentials; return the username on success, else raise AuthError."""
    username = normalize_username(username)
    with _db() as c:
        row = c.execute("SELECT salt, pw_hash FROM users WHERE username=?", (username,)).fetchone()
    # Run the hash even when the user is missing to avoid leaking timing.
    salt, expected = (row if row else (b"\x00" * 16, b""))
    candidate = _hash(password, salt)
    if not row or not hmac.compare_digest(candidate, expected):
        raise AuthError("Invalid username or password.")
    return username


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    with _db() as c:
        c.execute(
            "INSERT INTO sessions (token, username, created) VALUES (?,?,?)",
            (token, username, time.time()),
        )
    return token


def username_for_token(token: str | None):
    """Return the username for a valid, unexpired session token, else None."""
    if not token:
        return None
    with _db() as c:
        row = c.execute("SELECT username, created FROM sessions WHERE token=?", (token,)).fetchone()
    if not row:
        return None
    username, created = row
    if time.time() - created > _SESSION_TTL:
        delete_session(token)
        return None
    return username


def delete_session(token: str | None) -> None:
    if not token:
        return
    with _db() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))
