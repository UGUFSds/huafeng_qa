FROM python:3.10-slim-bookworm

WORKDIR /app

# 该项目依赖均通过 pip 安装，数据库驱动使用 psycopg2-binary，无需系统级依赖

# 安装Python依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码（尽量只复制需要的部分以减小镜像体积）
COPY scripts/ ./scripts/

# 运行脚本入口（可用 compose 覆盖传参）
ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/service.py"]