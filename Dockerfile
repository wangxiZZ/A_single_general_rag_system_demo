# syntax=docker/dockerfile:1
# =============================================================================
# 两阶段构建：builder 用 uv 解析并安装依赖，runtime 只保留 venv 与源码。
# 依赖层先于源码层复制，改代码不会触发依赖重装。
# =============================================================================

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) 只装第三方依赖（不含本项目），命中缓存即可秒级重建
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) 再装本项目（editable），后续改代码无需重装依赖
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# libgomp1：PyMuPDF 运行期依赖；curl：容器内手工排查依赖连通性用
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 上传文件与抽取图片的挂载点，提前建好并交给非 root 用户
RUN mkdir -p /data && chown app:app /data

USER app

EXPOSE 8000

CMD ["uvicorn", "rag_kb.main:app", "--host", "0.0.0.0", "--port", "8000"]
