// Neo4j 图谱 schema（幂等，启动时执行）。
//
// 唯一约束以 (kb_id, 业务 id) 为复合键，天然带库隔离；Entity 用 normalized 做唯一键，
// 同名不同写法（全角/括号说明）在应用层归一后落到同一个节点。
CREATE CONSTRAINT doc_key IF NOT EXISTS
FOR (d:Doc) REQUIRE (d.kb_id, d.doc_id) IS UNIQUE;

CREATE CONSTRAINT chunk_key IF NOT EXISTS
FOR (c:Chunk) REQUIRE (c.kb_id, c.chunk_id) IS UNIQUE;

CREATE CONSTRAINT entity_key IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.kb_id, e.normalized, e.type) IS UNIQUE;

CREATE INDEX doc_kb IF NOT EXISTS
FOR (d:Doc) ON (d.kb_id);

CREATE INDEX chunk_kb_doc IF NOT EXISTS
FOR (c:Chunk) ON (c.kb_id, c.doc_id);

CREATE INDEX entity_lookup IF NOT EXISTS
FOR (e:Entity) ON (e.kb_id, e.name);
