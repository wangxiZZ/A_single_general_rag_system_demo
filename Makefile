# 跨平台便利入口。Windows 无 make 时直接用 AGENTS.md §4 里的 docker compose 命令。
# 所有目标都是对 docker compose / uv 的薄封装，不含任何隐藏逻辑。

SHELL := /bin/bash
COMPOSE := docker compose
APP := ragkb-app

.DEFAULT_GOAL := help

.PHONY: help up down reset ps logs ready build test test-unit test-int lint fmt migrate demo bench shell-pg shell-neo4j

help: ## 显示所有目标
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## 启动全部服务并等待健康
	$(COMPOSE) up -d --build
	@echo "等待服务就绪（milvus 首次约 60-90s）..."
	@$(COMPOSE) ps
	@echo "就绪检查: curl -s localhost:$${APP_PORT:-8000}/readyz | python -m json.tool"

down: ## 停止服务（保留数据卷）
	$(COMPOSE) down

reset: ## 停止并清空数据卷（重建索引时用）
	$(COMPOSE) down -v

ps: ## 查看服务状态
	$(COMPOSE) ps

logs: ## 跟踪应用日志
	$(COMPOSE) logs -f app

ready: ## 打印 /readyz 结果
	@curl -s localhost:$${APP_PORT:-8000}/readyz | python -m json.tool

build: ## 仅重建应用镜像
	$(COMPOSE) build app

migrate: ## 在容器内执行数据库迁移
	$(COMPOSE) exec app rag-kb migrate

demo: ## 装载演示数据（宿主机执行，读取 tests/fixtures）
	uv run rag-kb demo

test: ## 跑全部测试（需服务已启动）
	uv run pytest -q

test-unit: ## 仅单测（无需容器）
	uv run pytest tests/unit -q

test-int: ## 仅集成测试（需 docker compose up）
	uv run pytest -m integration -q

lint: ## 静态检查
	uv run ruff check src tests mcp_server scripts
	uv run ruff format --check src tests

fmt: ## 自动格式化
	uv run ruff check --fix src tests mcp_server scripts
	uv run ruff format src tests mcp_server scripts

bench: ## 跑检索/回答 benchmark 并输出报告
	uv run python scripts/bench/run_bench.py

shell-pg: ## 进入 postgres psql
	$(COMPOSE) exec postgres psql -U $${PG_USER:-rag} -d $${PG_DB:-ragkb}

shell-neo4j: ## 进入 neo4j cypher-shell
	$(COMPOSE) exec neo4j cypher-shell -u $${NEO4J_USER:-neo4j} -p $${NEO4J_PASSWORD:-ragkb_neo4j_pw}
