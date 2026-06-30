"""Placeholder test to verify the test suite is configured correctly."""

from __future__ import annotations


def test_placeholder() -> None:
    """A simple test that always passes."""
    assert True


def test_imports() -> None:
    """Verify that core modules can be imported."""
    from oracle_flexcube_copilot import __version__  # noqa: F811

    assert __version__ == "0.1.0"