# 第一阶段：安装依赖
FROM python:3.9-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 第二阶段：生成最终镜像
FROM python:3.9-slim
WORKDIR /app

# 从第一阶段复制已安装的依赖
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY . .

# 设置环境变量，确保 Python 可以找到用户安装的包
ENV PATH=/root/.local/bin:$PATH

# 使用 uvicorn 启动 FastAPI 应用
CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]