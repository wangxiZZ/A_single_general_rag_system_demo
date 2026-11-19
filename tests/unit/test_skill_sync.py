"""Skill 源文件与安装目标必须一致。

`.claude/skills/` 是 Claude Code 的发现路径，`skills/` 是可提交、可评审的源文件。
两份内容一旦漂移，模型实际加载的规则和仓库里看到的规则就会不同；这个测试把
「记得重新复制」变成提交前的硬检查。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "skills" / "rag-kb" / "SKILL.md"
INSTALLED = REPO_ROOT / ".claude" / "skills" / "rag-kb" / "SKILL.md"


def test_installed_skill_matches_source() -> None:
    assert SOURCE.is_file(), f"缺少 Skill 源文件：{SOURCE}"
    assert INSTALLED.is_file(), f"缺少安装副本：{INSTALLED}（见 docs/mcp-setup.md §4）"
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
