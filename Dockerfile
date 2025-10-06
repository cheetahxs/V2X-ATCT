# Dockerfile
# FROM continuumio/miniconda3:latest
# # FROM registry.cn-hangzhou.aliyuncs.com/aliyun-conda/miniconda3:latest

# # 设置工作目录（通用路径）
# WORKDIR /app

# # 复制 environment.yml
# COPY environment.yml .

# # 设置 Conda 环境变量（可选）
# ENV CONDA_DEFAULT_ENV=v2x-atct

# # 关键：先设置 SHELL，让后续 RUN 在 conda 环境中执行
# SHELL ["conda", "run", "-n", "v2x-atct", "/bin/bash", "-c"]

# # 创建 Conda 环境
# RUN conda env create -f environment.yml && \
#     conda clean -a -y

# # 恢复默认 SHELL，避免影响 COPY 等指令
# SHELL ["/bin/bash", "-c"]

# # 复制项目文件
# COPY . .

# # 暴露端口
# EXPOSE 5000

# # 启动命令
# ENTRYPOINT ["conda", "run", "-n", "v2x-atct", "python"]
# CMD ["Visualization/app.py"]


FROM continuumio/miniconda3:4.12.0

# FROM continuumio/miniconda3:latest
# 安装系统依赖（关键！）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         libgomp1 \
#         libgl1 \
#         libglib2.0-0 \
#         libx11-6 \
#         libsm6 \
#         libxext6 \
#         libxrender-dev \
#         # ============ 添加开发依赖 ============
#         build-essential \
#         libssl-dev \
#         libffi-dev \
#         libcrypt-dev \          # ✅ 替换 libxcrypt-dev 为 libcrypt-dev
#         python3-dev \
#         # ========================================
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*



WORKDIR /app

# 复制环境文件
COPY environment.yml .

# 在 base 环境中安装依赖（推荐）
RUN conda env update -n base -f environment.yml && \
    conda clean -a -y

# 复制其他代码
COPY . .

# ENV PYTHONPATH="/app:${PYTHONPATH}"
# 推荐：安全地将 /app 添加到 PYTHONPATH 前面
ENV PYTHONPATH=/app:${PYTHONPATH:-}

RUN cd /app/V2X-ATCT/MultiTest-master && \
    python build_script.py

# 设置默认命令
CMD ["python", "Visualization/app.py"]