# 滑冰馆租赁报修分流

清屿滑冰馆的租赁柜台收集冰鞋报修短句。后端按历史班次拟合字符 TF-IDF 与 Complement Naive Bayes，将意见明确的工单分给冰刀刃磨台、鞋体整备台和固定件检修台，含糊工单由柜台补问。数据口径和交班验收见 [BUSINESS.md](BUSINESS.md)。

| 模块 | 工作内容 |
|---|---|
| 班次台账 | 导入 UTF-8 CSV，检查工单号、标签与班次边界，冻结上传快照 |
| 分派运行 | 使用独立的历史训练、整班留出和待修分区，由 worker 计算 |
| 分站报告 | 查看每班和各类别的留出表现、先验基线、逐单分数及待修队列 |
| 交班签核 | 记录签核决定与审计；通过后开放正式 CSV 和查阅员结果访问 |

## 启动

```sh
cp .env.example .env
docker compose build backend
docker compose up -d --no-build --wait
docker compose exec -T backend python -m scripts.seed_demo
```

API 绑定 `127.0.0.1:19433`，OpenAPI 文件位于 `http://127.0.0.1:19433/api/openapi.json`。MySQL 8.0 宿主端口为 `17933`。Compose 的四个服务为 `backend`、`worker`、`db` 和 `redis`。SQLAlchemy 将来源、运行、签核及审计写入 MySQL，Redis 用于唤醒 worker。`docker compose down` 会保留数据卷。

| 环境变量 | 用途 |
|---|---|
| `COMPOSE_PROJECT_NAME` | 默认 `yangyang-skate-text` |
| `BACKEND_PORT` / `DB_PORT` | 宿主 API / MySQL 端口 |
| `SEED_DEMO` | 控制启动时自动写入演示数据，默认 `true` |
| `JWT_SECRET` | 本地签名密钥，部署时按运行环境配置 |
| `DATABASE_URL` / `REDIS_URL` | Compose 注入的数据库与队列地址 |

| 本地演示账号 | 口令 | 可执行动作 |
|---|---|---|
| `analyst@demo.local` | `Demo-2611!` | 导入台账，发起运行，查看结果，导出获批 CSV |
| `admin@demo.local` | `Demo-2611!` | 导入台账，发起运行，签核交班，查看审计，导出获批 CSV |
| `viewer@demo.local` | `Demo-2611!` | 查看来源与获批结果，下载获批 CSV |

## API

先调用 `POST /api/auth/login`，提交 `email` 和 `password`，再为需要认证的请求附上 `Authorization: Bearer <access_token>`。

| 操作 | 接口 | 请求或结果 |
|---|---|---|
| 核对服务与项目规格 | `GET /api/health`、`GET /api/spec` | 健康状态、列名、运行参数与模块名称 |
| 处理台账 | `GET /api/datasets`、`POST /api/datasets` | 导入请求为 `{"name":"台账.csv","csv":"完整 CSV 文本"}` |
| 预览来源 | `GET /api/datasets/{dataset_id}` | `offset` 和 `limit` 分页 |
| 提交和查看运行 | `POST /api/runs`、`GET /api/runs/{run_id}` | 提交 `label`、`dataset_id` 和可选的 `settings`，查询运行状态和结果 |
| 签核交班 | `POST /api/runs/{run_id}/review` | 提交 `decision` 和 `note`，签核通过用 `approved` |
| 下载结果 | `GET /api/runs/{run_id}/export/{key}` | `pending_routes`、`holdout_by_shift`、`holdout_by_label`、`holdout_predictions`、`training_profile` |
| 查审计记录 | `GET /api/audit` | 管理员近期操作记录 |

## 本地开发

`backend/src/domain/` 包含输入规则、班次夹具、文本特征、分类器、评估与报表，使用 Python 3.11 和 scikit-learn 1.6.1。`backend/src/routes/`、`services/` 和 `models/` 提供 FastAPI 工作流和持久化。`backend/scripts/seed_demo.py` 按账号和文件名初始化演示台账。`backend/src/constants/workflow.py` 定义 JWT 角色、运行状态和签核决定；分区与故障类别列于 `backend/src/domain/settings.py`。第三方依赖采用各自许可证。

安装 `backend/requirements.txt` 后，从项目根目录运行算法 oracle。校验读取项目的真实 CSV 工具以及业务入口，通过标准库独立复算模型分数。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend \
  python ../../test-artifacts/yangyang-skate-text/tests/oracle.py
```
