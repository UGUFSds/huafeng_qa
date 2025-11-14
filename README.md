# Huafeng QA

本项目是面向工业数据的问答与查询系统，集成了多源路由（SQL 与 CSV）、澄清交互、链路评估与本地可观测（Phoenix UI）。重点在速度与稳定性优化，同时支持本地数据库与离线数据源。

## 快速开始
- 环境准备
  - 安装依赖：`python -m pip install -r requirements.txt`
  - 本地数据库（推荐 Postgres with Docker）：
    - 启动数据库：`docker compose up -d db`
    - 导入数据（初次）：`docker compose run --rm db-seed`
- `.env` 样例（本地开发）
  - `POSTGRES_HOST=127.0.0.1`
  - `POSTGRES_PORT=5433`
  - `POSTGRES_DB=app_db`
  - `POSTGRES_USER=postgres`
  - `POSTGRES_PASSWORD=postgres`
  - `POSTGRES_SSLMODE=disable`
  - `LLM_BASE_URL=https://api.deepseek.com`
  - `LLM_API_KEY=<your_key>`
- 运行一次查询
  - 列表查询：`python scripts/service.py --question "列出数据库中的表名，显示前10个" --lang zh`
  - 带澄清查询：`python scripts/service.py --question "查一下RA2粘度在9.17日下午6点的数据" --lang zh --clarify "1" --report-dir reports`

## Phoenix UI（可观测与评估）
- 启用方式
  - 启动 UI：`phoenix serve`（浏览器访问 `http://localhost:6006/`）
  - 环境启用：运行查询时设置 `PHOENIX_ENABLED=1` 与 `PHOENIX_ENDPOINT=http://localhost:6006/v1/traces`
  - 示例：`PHOENIX_ENABLED=1 PHOENIX_ENDPOINT=http://localhost:6006/v1/traces python scripts/service.py --question "测试" --lang zh --report-dir reports`
- 集成功能
  - 每条查询链路自动上报 spans（步骤耗时、返回规模、评估属性）
  - `post_query_eval` span 附带质量评估（相关性、完整性、清晰度与简短说明）
  - 本地报告 JSON（`reports/query_report_*.json`）包含 `steps/cost/evals/source_coverage` 字段
- 常见预览错误（预览沙箱阻断外链）
  - 预览窗口可能阻断外部资源（JS/CSS/视频），不影响核心链路；使用系统浏览器访问 `http://localhost:6006/`

## 路由与查询链路
- 数据源与顺序（模型自主）：SQL（历史/报警）、CSV（点位主数据）。模型根据问题与数据源关系自主决定先查哪个来源；当 SQL 阶段出现需要 CSV 的意图时，系统追加一次 CSV 步骤并基于候选重写 SQL（仅触发一次）。
- 澄清与“仅查所选”
  - 当候选过多时返回澄清选项；选择后仅查询被选点位（数据库仅暴露所选目标表），避免自动扩展到近邻表。
  - 多选场景下，数据库侧仅可见多选集合内的目标表，模型自然比较多个点位。
- 终端输出（结构化）
  - 计划说明前缀显示“所选点位”标签
  - 工具调用结构化展示：SQL 的“目标表”和“时间窗”
  - 跨源意图统一输出为“路由提示”，降低无效工具噪声

## 稳定性与性能优化
- 数据库稳定性
  - 连接池与探活：`pool_pre_ping=True`
  - 长连接回收：`pool_recycle=1800`
  - 资源控制：`pool_size=5`、`max_overflow=10`
  - 驱动超时与语句超时：`connect_timeout=5`、`statement_timeout=5000ms`
- 并发调度
  - 在已有事件循环场景使用线程池并发执行源步骤，减少退化串行；在非运行状态使用 asyncio 并发
- 动态跨源回补（幂等）
  - SQL 阶段发现 CSV 意图时，自动追加一次 CSV 步骤并重写 SQL；仅触发一次，避免重复回补
- 缓存与自省
  - 探测缓存默认 `120s`，重写缓存默认 `600s`，减少重复自省与提示开销
- CSV 回退筛选（轻量）
  - 关键词去重并限前 5 个；优先匹配 `point_name/code/table_name`；返回上限 20 行

## 配置项说明（摘）
- LLM
  - `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`LLM_REQUEST_TIMEOUT`、`LLM_MAX_TOKENS`
- Postgres
  - `POSTGRES_*`（`HOST/PORT/DB/USER/PASSWORD/SSLMODE`）
  - `DB_URI` 自动拼接（支持 `sslmode` 查询参数）
- 路由与缓存
  - `ROUTER_PARALLEL_EXECUTE` 并发开关
  - `ROUTER_PROBE_CACHE_SECONDS`、`ROUTER_REWRITE_CACHE_SECONDS`、`ROUTER_RESULT_CACHE_SECONDS`
  - `ROUTER_STRICT_AFTER_CLARIFICATION`（澄清后严格命中，仅暴露所选目标表）
- CSV
  - `CSV_AGENT_SECOND_PASS`（二次总结）
  - `CSV_FALLBACK_AUTO_FILTER`（自动回退筛选）
- 评估与报告
  - `EVALS_REPORT_DIR` 报告目录；`PHOENIX_*`（`ENABLED/ENDPOINT/PROJECT_NAME`）

## 常见问题
- invalid_tool csv_lookup
  - 出现于 SQL 阶段尝试跨源调用；系统会自动追加 CSV 步骤，终端显示为“路由提示”，无需人工干预
- “访问受限”提示
  - 澄清后仅查所选；若模型尝试未选表将被拒绝。多选以扩展对比范围；或关闭严格模式做扩展探索
- Phoenix 预览外链报错
  - 预览沙箱阻断外链 JS/CSS/视频；使用系统浏览器访问即可

## 目录结构（节选）
- `app/config/settings.py` 配置与环境变量读取
- `app/sources/sql.py` SQL 数据源、连接池与超时、年份守护、可用表过滤
- `app/sources/csv.py` CSV 数据源、二次总结、回退筛选
- `app/router/orchestrator.py` 路由执行、并发调度、跨源回补、缓存
- `app/callbacks/console.py` 终端输出格式化
- `app/observability/phoenix.py` Phoenix 集成与批处理导出
- `scripts/service.py` 服务入口、一次性与交互式查询、报告写入

## 开发与运行建议
- 本地数据库优先：`sslmode=disable`；生产建议启用 TLS 与更严格审计
- 慢查询判定：默认 `statement_timeout=5000ms`，可根据需求提高
- 报告落盘：设置 `--report-dir reports` 生成 JSON，便于回归检查与离线分析

## 推送与版本
- 提交说明聚焦：监控与评估、路由与稳定性、CSV 与并发优化、DB 稳定性、终端与报告
- 推送到主分支：`git add . && git commit -m "docs: add README and usage" && git push origin main`

如需将 README 拆分为“用户指南/运维手册/二次开发”三个文档，请告知，我可以按上述结构为你生成更细分的文档。
