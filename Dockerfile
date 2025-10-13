# 基于包含 CUDA 的 Python 镜像（适配项目中的 GPU 依赖，如 nvidia-cuda 相关包）
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖（含图形界面所需库）
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libgtk2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目所有文件
COPY . .

# 设置环境变量（支持 GUI 显示）
ENV DISPLAY=:0

# 启动命令（运行主程序）
CMD ["python3", "a_demonstration.py"]