"""种子实体链接单测。"""

from __future__ import annotations

from uuid import UUID, uuid4

from rag_kb.application.entity_linking import bigrams, link_seed_entities, query_segments
from rag_kb.domain.models import Entity

KB_ID = uuid4()


class FakeGraphStore:
    def __init__(self, exact: list[Entity], fuzzy: list[Entity] | None = None) -> None:
        self.exact = exact
        self.fuzzy = fuzzy or []
        self.calls: list[tuple[list[str], bool]] = []

    async def candidate_entities(
        self, kb_id: UUID, terms, limit: int, *, contains: bool = False
    ) -> list[Entity]:
        self.calls.append((list(terms), contains))
        return self.fuzzy if contains else self.exact


def _entity(entity_id: str, name: str) -> Entity:
    return Entity(
        id=entity_id,
        kb_id=KB_ID,
        name=name,
        normalized=name,
        type="条款",
    )


def test_query_segments_and_bigrams() -> None:
    assert query_segments("年假政策，有哪些？") == ["年假政策", "有哪些"]
    assert "年假" in bigrams(["年假政策"])


async def test_exact_hit_skips_fuzzy() -> None:
    entity = _entity("e1", "年假政策")
    store = FakeGraphStore(exact=[entity])

    result = await link_seed_entities(store, KB_ID, "年假政策")  # type: ignore[arg-type]

    assert result == [entity]
    assert store.calls == [(["年假政策"], False)]


async def test_fuzzy_fallback_when_exact_empty() -> None:
    entity = _entity("e1", "年假政策")
    store = FakeGraphStore(exact=[], fuzzy=[entity])

    result = await link_seed_entities(store, KB_ID, "年假政策")  # type: ignore[arg-type]

    assert result == [entity]
    assert len(store.calls) == 2
    assert store.calls[1][1] is True
    assert "年假" in store.calls[1][0]


async def test_fuzzy_results_are_deduplicated_by_id() -> None:
    entity = _entity("e1", "年假政策")
    store = FakeGraphStore(exact=[], fuzzy=[entity, entity, _entity("e2", "带薪年假")])

    result = await link_seed_entities(store, KB_ID, "年假政策")  # type: ignore[arg-type]

    assert [item.id for item in result] == ["e1", "e2"]


async def test_no_terms_returns_empty_without_query() -> None:
    store = FakeGraphStore(exact=[])
    assert await link_seed_entities(store, KB_ID, "，。！") == []  # type: ignore[arg-type]
    assert store.calls == []
