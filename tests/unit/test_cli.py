"""CLI 入口的轻量单测。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_kb import cli


def test_migrate_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(pg_dsn="dsn"))
    monkeypatch.setattr(cli, "apply_migrations", lambda dsn: ["002_document_file_path"])

    assert cli.main(["migrate"]) == 0
    assert "002_document_file_path" in capsys.readouterr().out


def test_help_lists_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "migrate" in output
    assert "demo" in output
    assert "serve" in output
