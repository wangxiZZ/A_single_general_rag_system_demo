"""通用 RAG 知识库服务。

分层约定（见 AGENTS.md，依赖禁止回指）：
    api / web / worker  →  application  →  domain
    infra               实现 domain 中定义的端口，由 main.py 在启动期注入

domain 层不得 import infra 层，保证核心业务与具体存储解耦。
"""

__version__ = "0.1.0"
