"""全局配置：唯一的环境变量读取点。

设计原因
--------
所有外部依赖的地址、密钥、可调参数都集中在 Settings 中，其余模块只接收已解析的
配置对象，不自行读 env。这样：

1. 测试可以用 ``Settings(...)`` 直接构造，无需污染进程环境；
2. 配置项一改，影响面在类型检查期就能看到；
3. ``.env.example`` 与 Settings 字段一一对应，验收时"填完 key 就能起"有据可查。

取舍：不引入多环境配置文件（dev/prod yaml），单机 DEMO 场景用 env + 默认值足够，
多一层配置文件只会让"启动步骤"更难写清楚。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行期配置。字段默认值与 docker-compose 里的服务名保持一致。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 应用 ----
    app_name: str = "rag-kb"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    # 上传文件、抽出的图片、原始解析结果的落盘根目录。
    # 默认相对路径便于本机直跑；容器内由 compose 覆盖为挂载点 /data。
    data_dir: Path = Path("./data")

    # ---- PostgreSQL：业务元数据 + 词法召回 ----
    pg_dsn: str = "postgresql://rag:rag@postgres:5432/ragkb"
    pg_pool_min: int = 1
    pg_pool_max: int = 8

    # ---- Milvus：稠密向量召回 ----
    milvus_uri: str = "http://milvus:19530"
    milvus_token: str = ""
    milvus_collection: str = "chunk_vec"
    # 向量维度必须与 embedding 模型一致；启动时与 collection 实际维度比对，不一致直接
    # fail-fast，避免"能启动但检索恒空"这类难定位的故障。
    embedding_dim: int = 1024

    # ---- Neo4j：图谱召回 ----
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ragkb_neo4j_pw"
    neo4j_database: str = "neo4j"

    # ---- LLM：生成与图谱抽取 ----
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout_s: int = 90
    llm_max_concurrency: int = 4

    # ---- Embedding ----
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 16
    embed_timeout_s: int = 60

    # ---- 多模态（图片描述）----
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = "qwen-vl-max"
    vlm_timeout_s: int = 90
    # 关闭时图片仍会入索引，content 退化为"[图片] 无法描述"占位，便于断网调试。
    vlm_enabled: bool = True

    # ---- 分块参数（题目要求方案可解释，故全部可配）----
    chunk_target_tokens: int = 320
    chunk_overlap_ratio: float = 0.15
    parent_max_tokens: int = 1200
    # 中文按字符估 token 的系数：粗略 1 token ≈ 1.5 汉字。仅用于分块长度控制，
    # 不追求精确计费口径。
    chars_per_token: float = 1.5

    # ---- 检索参数 ----
    retrieval_top_k: int = 20
    fusion_rrf_k: int = 60
    rerank_enabled: bool = False
    graph_max_hops: int = 2
    graph_seed_limit: int = 8

    # ---- 上传限制 ----
    max_upload_mb: int = 50
    allowed_mime: tuple[str, ...] = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.image_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def model_api_configured(self) -> dict[str, bool]:
        """判断各类模型能力是否已配置。

        /readyz 用它来区分"服务本身不可用"与"模型未配置"。后者不阻塞启动——
        否则验收时忘填一个 key 就连健康检查都过不去，反而更难排查。
        """
        return {
            "llm": bool(self.llm_api_key),
            "embedding": bool(self.embed_api_key),
            "vlm": bool(self.vlm_api_key and self.vlm_enabled),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。uvicorn 多 worker 时每个进程各持一份，配置本身无状态，安全。"""
    return Settings()
