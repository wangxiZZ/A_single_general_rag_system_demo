-- 001_init.sql — 业务元数据与词法索引的初始 schema。
--
-- 设计取舍：
-- 1. 不用 ORM，DDL 手写；迁移版本由 schema_migration 表记录，执行器按文件名排序。
-- 2. 状态/类型用 text + CHECK，而不是 PostgreSQL 原生 ENUM：加值只需改约束，
--    不必 ALTER TYPE，回滚也更简单。
-- 3. chunk.parent_id 自引用外键设为 DEFERRABLE：同一事务内先插父块再插子块，
--    批量写入时不必关心语句顺序。
-- 4. chunk_lex.tsv 用生成列，避免应用层维护两份词法文本。

CREATE TABLE IF NOT EXISTS kb (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL,
    description text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id           uuid NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    filename        text NOT NULL,
    mime            text NOT NULL,
    sha256          char(64) NOT NULL,
    size            bigint NOT NULL CHECK (size >= 0),
    status          text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'parsing', 'chunked', 'indexing', 'ready', 'failed')),
    error           text,
    page_count      integer,
    chunk_count     integer NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    chunker_version text NOT NULL DEFAULT '',
    embed_model     text NOT NULL DEFAULT '',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_document_kb_sha UNIQUE (kb_id, sha256)
);

CREATE INDEX IF NOT EXISTS idx_document_kb_created
    ON document (kb_id, created_at DESC);

CREATE TABLE IF NOT EXISTS doc_image (
    id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id     uuid NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    doc_id    uuid NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    seq       integer NOT NULL CHECK (seq >= 0),
    page_no   integer,
    bbox      jsonb,
    file_path text NOT NULL,
    caption   text,
    CONSTRAINT uq_doc_image_doc_seq UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_doc_image_doc ON doc_image (doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_image_kb ON doc_image (kb_id);

CREATE TABLE IF NOT EXISTS chunk (
    id              varchar(32) PRIMARY KEY,
    kb_id           uuid NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    doc_id          uuid NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    parent_id       varchar(32) REFERENCES chunk(id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    level           text NOT NULL CHECK (level IN ('parent', 'child')),
    kind            text NOT NULL CHECK (kind IN ('text', 'image', 'table')),
    content         text NOT NULL,
    heading_path    text[] NOT NULL DEFAULT '{}',
    page_no         integer,
    page_to         integer,
    bbox            jsonb,
    char_start      integer NOT NULL CHECK (char_start >= 0),
    char_end        integer NOT NULL CHECK (char_end >= 0),
    seq             integer NOT NULL CHECK (seq >= 0),
    token_count     integer NOT NULL CHECK (token_count >= 0),
    image_id        uuid REFERENCES doc_image(id) ON DELETE SET NULL,
    meta            jsonb NOT NULL DEFAULT '{}'::jsonb,
    chunker_version text NOT NULL,
    CONSTRAINT ck_chunk_char_range CHECK (char_end >= char_start),
    CONSTRAINT uq_chunk_doc_seq UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunk_kb_doc ON chunk (kb_id, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_parent ON chunk (parent_id);
CREATE INDEX IF NOT EXISTS idx_chunk_image ON chunk (image_id);

CREATE TABLE IF NOT EXISTS chunk_lex (
    chunk_id   varchar(32) PRIMARY KEY REFERENCES chunk(id) ON DELETE CASCADE,
    lex_tokens text NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('simple', lex_tokens)) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunk_lex_tsv ON chunk_lex USING gin (tsv);

CREATE TABLE IF NOT EXISTS ingest_job (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id     uuid NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    stage      text NOT NULL DEFAULT 'parse',
    status     text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    attempts   integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    error      text,
    timings    jsonb NOT NULL DEFAULT '{}'::jsonb,
    claimed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ingest_job_status
    ON ingest_job (status, created_at);

-- 同一文档最多只有一个待处理/执行中的任务，防止重复上传或重试造成并发摄取。
CREATE UNIQUE INDEX IF NOT EXISTS uq_ingest_job_active
    ON ingest_job (doc_id) WHERE status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS query_log (
    id         bigserial PRIMARY KEY,
    kb_id      uuid NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    mode       text NOT NULL
        CHECK (mode IN ('vector', 'keyword', 'graph', 'hybrid', 'hybrid_weighted')),
    query      text NOT NULL,
    latency_ms double precision NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
    hit_ids    jsonb NOT NULL DEFAULT '[]'::jsonb,
    channels   jsonb NOT NULL DEFAULT '{}'::jsonb,
    rerank     boolean NOT NULL DEFAULT false,
    tokens     integer,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_log_kb_created
    ON query_log (kb_id, created_at DESC);
