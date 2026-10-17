# 代码知识库问答系统 —— 生产镜像
#
# 镜像**包含**向量索引（storage/，约 51MB），**不含**语料（data/）：
#   - 索引随镜像走，云端部署不必在容器里重跑 ingest（重跑要花 embedding 费用 + 8 分钟），
#     代价是镜像大 51MB、且重建索引后要重新构建镜像；
#   - data/ 是第三方官方文档，用 fetch_docs.py 能重抓，没必要占镜像体积
#     （检索只查索引，运行时不需要原始语料）。
#
# 构建：docker build -t codebase-copilot .
# 运行：docker run -p 8000:8000 --env-file .env codebase-copilot

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # 容器里没有桌面环境，别去调 webbrowser 开浏览器
    WBRAG_NO_BROWSER=1

WORKDIR /app

# 先单独拷贝依赖清单再安装：改业务代码时这一层能命中缓存，不用重装几百兆依赖。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 必须监听 0.0.0.0：只听 127.0.0.1 的话容器外（宿主/负载均衡）访问不到。
# 这两个变量会被 pydantic-settings 直接读进 settings.api_host / api_port。
ENV API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

# 用 python 自带 urllib 探活，省掉在镜像里装 curl。
# start-period 给足 60s：启动时要从磁盘读近万片段的 FAISS 索引。
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "app.api"]
