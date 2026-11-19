"""配置三处同步自检。

容器不读 `.env`，compose 的 `environment` 是唯一透传通道；任何 Settings 字段漏传都会
表现为"改了配置不生效"。这里用机器校验代替人工核对。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from rag_kb.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

_ENV_LINE_RE = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=")


def _setting_names() -> set[str]:
    return {name.upper() for name in Settings.model_fields}


def _compose_app_env() -> set[str]:
    data = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    environment = data["services"]["app"]["environment"]
    assert isinstance(environment, dict), "compose app.environment 必须是映射"
    return {key.upper() for key in environment}


def _env_example_names() -> set[str]:
    names: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if match := _ENV_LINE_RE.match(line):
            names.add(match.group(1))
    return names


def test_every_setting_is_passed_to_app_container() -> None:
    missing = _setting_names() - _compose_app_env()
    assert missing == set(), f"以下 Settings 字段没有在 compose 中透传: {sorted(missing)}"


def test_compose_has_no_unknown_app_settings() -> None:
    unknown = _compose_app_env() - _setting_names()
    assert unknown == set(), f"compose 透传了 Settings 不认识的变量: {sorted(unknown)}"


def test_env_example_documents_every_setting() -> None:
    missing = _setting_names() - _env_example_names()
    assert missing == set(), f".env.example 缺少字段说明: {sorted(missing)}"
