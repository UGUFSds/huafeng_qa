FROM python:3.10-slim

WORKDIR /app

# 基础系统依赖（psycopg2、curl等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
  && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码（尽量只复制需要的部分以减小镜像体积）
COPY scripts/ ./scripts/
COPY sql_schema_qa.py ./

# 运行脚本入口（可用 compose 覆盖传参）
ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/huafeng_service.py"]