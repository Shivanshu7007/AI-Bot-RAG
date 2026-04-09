"""Unit tests for app/core/security.py"""
import pytest
from fastapi import HTTPException
from unittest.mock import patch


class _Settings:
    SERVICE_API_KEY = "test-secret-key-abc123"


with patch("app.core.config.settings", _Settings()):
    from app.core.security import verify_api_key


def test_valid_key_passes():
    result = verify_api_key(x_api_key="test-secret-key-abc123")
    assert result == "test-secret-key-abc123"


def test_missing_key_raises_401():
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key=None)
    assert exc.value.status_code == 401


def test_wrong_key_raises_403():
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key="wrong-key")
    assert exc.value.status_code == 403


def test_empty_string_key_raises_403():
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key="")
    assert exc.value.status_code == 403


def test_unconfigured_service_key_raises_500():
    class _NoKey:
        SERVICE_API_KEY = None

    with patch("app.core.security.settings", _NoKey()):
        with pytest.raises(HTTPException) as exc:
            verify_api_key(x_api_key="anything")
        assert exc.value.status_code == 500
