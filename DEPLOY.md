# 新服务器 Docker 部署文档

## 一、环境要求

- Docker Engine 24+ + Docker Compose
- Git
- （可选，GPU需用）NVIDIA Container Toolkit + NVIDIA 驱动

## 二、部署步骤

### 1. 拉取代码

```bash
git clone https://github.com/823680560/san.git
cd san
```

### 2. 放入本地数据

> 以下 `data/` 为业务数据（需手动准备），`models/` 为嵌入模型和精排模型权重（需提前下载），`knowledge_base/` 为向量索引（首次运行时自动构建）。

```
san/
├── data/
│   ├── law/                     # 法规资料（PDF/JSON）
│   └── project/                 # 环卫项目数据（Excel）
├── knowledge_base/              # 向量索引（首次运行时自动构建）
└── models/                      # HuggingFace 本地模型权重
    ├── bge-m3/                  # 嵌入模型（需提前下载）
    └── bge-reranker-v2-m3/      # 精排模型（需提前下载）
```

**下载模型**（在 `san/` 项目根目录下执行，推荐选择将本地已下载好的模型文件传送到san文件夹下的models文件夹中，这样可以避免网络不稳定造成的下载失败）：

方式一：用 Docker 临时容器下载（宿主机无需安装 Python）

```bash
docker run --rm \
  -v $(pwd)/models:/models \
  -e HF_ENDPOINT=https://hf-mirror.com \
  python:3.11-slim \
  sh -c "pip install -q huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    hf download BAAI/bge-m3 --local-dir /models/bge-m3 && \
    hf download BAAI/bge-reranker-v2-m3 --local-dir /models/bge-reranker-v2-m3"
```

方式二：宿主机有 Python 时直接下载

```bash
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
hf download BAAI/bge-m3 --local-dir models/bge-m3
hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

> 每个模型约 2.2 GB，支持断点续传（中断后重复执行即可）。
> `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像加速下载，海外服务器可去掉此行。
> 下载完成后确认 `models/bge-m3/` 和 `models/bge-reranker-v2-m3/` 两个目录存在且包含模型文件。

### 3. 配置环境变量

```bash
cd docker
cp .env.example .env
```

编辑 `.env`，根据 LLM 方案二选一：

**方案 A：使用 DeepSeek API（默认）**

```ini
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-你的key
LLM_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_SERVICE=huggingface
```

**方案 B：使用 Ollama 本地模型**

```ini
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=ollama
LLM_BASE_URL=http://ollama:11434/v1
EMBEDDING_SERVICE=huggingface
```

### 4. 启动服务

```bash
cd docker
docker compose up -d
```

> **Ollama 默认不启动**，使用在线 API 模式。如需切换到本地模型，参考下方方案 B。

启动后约 2-3 分钟，访问 `http://服务器IP:8000`。

### 5. 切换到 Ollama 本地模型

> **前提**：使用 Ollama 需要用户自行安装 Ollama 并启动服务。执行 `docker compose --profile ollama up -d` 时 Docker 会自动拉取 Ollama 容器镜像，但 Ollama 内部的 LLM 对话模型（如 `qwen2.5:7b`）仍需手动执行 `ollama pull` 下载。

如果要用本地模型而非在线 API，额外执行以下步骤：

① 拉取 Ollama 镜像并拉取对话模型（仅首次需执行）：

```bash
# 启动 Ollama 容器（指定 ollama profile）
docker compose --profile ollama up -d

# 进入容器拉取对话模型
docker exec san-ollama ollama pull qwen2.5:7b

# 确认已拉取
docker exec san-ollama ollama list
```

② 修改 `.env`，将 LLM 配置改为方案 B（Ollama 本地模型），重启服务。

③ 之后正常启动也需加 profile：

```bash
docker compose --profile ollama up -d
```

### 6. 构建知识库

访问 `http://服务器IP:8000`，按顺序操作：

| 页面 | 功能 |
|------|------|
| `/upload` | 上传数据文件（Excel / PDF / JSON） |
| `/vectorize` | 对文件分片并构建 FAISS 向量索引 |
| `/chat` | 选择知识库开始对话 |

## 三、模型加载方式

| 模型 | 用途 | 加载方式 |
|------|------|---------|
| `models/bge-m3/` | 文本嵌入（向量化） | 宿主机预下载 → 容器直接本地加载 |
| `models/bge-reranker-v2-m3/` | 检索结果精排 | 宿主机预下载 → 容器直接本地加载 |
| Ollama 对话模型 | 生成回答 | Ollama 服务调用 |

嵌入和精排模型通过 Docker volume 挂载到容器内 `/app/models/`。容器启动时检测本地路径存在，直接加载，不走网络下载。

## 四、常用命令

```bash
# 启动服务（默认在线 API 模式）
docker compose up -d

# 启动服务（Ollama 本地模型模式）
docker compose --profile ollama up -d

# GPU 环境启动（需 NVIDIA Container Toolkit）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile ollama up -d

# 查看日志
docker compose logs -f app        # Web 日志
docker compose logs -f ollama     # Ollama 日志（需先 profile 启动）

# 重启（修改配置后）
docker compose down && docker compose up -d

# 重新构建镜像（依赖变更时）
docker compose build --no-cache app

# 停止服务
docker compose down
```

## 五、注意事项

- `data/`、`knowledge_base/`、`models/` 通过 volume 挂载到容器，修改本地文件后无需重构建镜像
- 嵌入和精排模型（bge-m3 / bge-reranker）已提前下载到 `models/` 目录，容器启动时直接本地加载，无需联网
- Ollama 仅用于运行对话模型（LLM），嵌入模型走本地加载，更快且无需网络
- Reranker 必须用 HuggingFace 本地模型（Ollama 不支持 CrossEncoder）
- 首次启动因下载 Python 依赖较多（镜像构建阶段），一般需要 3-5 分钟；后续启动约 30-60 秒
- `LLM_*` 变量中 `http://ollama:11434/v1` 的 `ollama` 是 Docker Compose 服务名，容器内自动解析
- **Ollama 服务默认不启动**：`profiles: [ollama]` 确保只有显式 `--profile ollama` 时才拉取并启动。默认线上 API 模式无需下载 Ollama 镜像，开机更快。
- **GPU 加速**：`docker-compose.gpu.yml` 是 override 文件，CPU 环境只需 `docker compose up -d`，GPU 环境加 `-f` 参数合并使用。
  宿主机需提前安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
