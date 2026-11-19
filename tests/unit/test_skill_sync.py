"""Skill 源文件与安装副本一致性检查。

`skills/rag-kb/SKILL.md` 是唯一入库的源文件；`.claude/skills/rag-kb/SKILL.md` 是按需复制
到本机的安装副本（不入库，见 `.gitignore`）。这里只在本机装了副本时校验两份内容一致，
避免「实际加载的规则」和仓库里的源文件悄悄漂移；没安装时跳过，不影响新克隆跑单测。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "skills" / "rag-kb" / "SKILL.md"
INSTALLED = REPO_ROOT / ".claude" / "skills" / "rag-kb" / "SKILL.md"


def test_installed_skill_matches_source_when_present() -> None:
    assert SOURCE.is_file(), f"缺少 Skill 源文件：{SOURCE}"
    if not INSTALLED.is_file():
        pytest.skip("未安装 Skill 副本；安装方法见 docs/mcp-setup.md §4")
    assert INSTALLED.read_bytes() == SOURCE.read_bytes(), (
        "Skill 源文件与 .claude/skills 安装副本不一致；"
        "修改 skills/rag-kb/SKILL.md 后重新复制到 .claude/skills/rag-kb/"
    )


def test_skill_frontmatter_declares_name() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "Skill 必须以 YAML frontmatter 开头"
    frontmatter = text.split("---", 2)[1]
    assert "name: rag-kb" in frontmatter
    assert "description:" in frontmatter
