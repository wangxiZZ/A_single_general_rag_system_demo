-- 002_document_file_path.sql — 记录上传件的相对路径。
--
-- 为什么现在补：摄取 worker 需要从 Storage 读回原文件；路径在 M1 上传时已生成，
-- 但没有落库。存相对路径（uploads/<kb>/<doc>/<file>）保证宿主与容器可移植。

ALTER TABLE document
    ADD COLUMN IF NOT EXISTS file_path text NOT NULL DEFAULT '';
