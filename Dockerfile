
LABEL authors="Phyven"

# 基础镜像：NVIDIA CUDA 11.8 + Ubuntu 20.04（匹配本地CUDA版本，减少冲突）
# 选择runtime版本，包含CUDA运行时但体积较小
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# 设置非交互模式，避免apt安装时弹出配置窗口
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖和Python 3.8（匹配本地Python版本）
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3.8-venv \
    # 安装编译依赖（部分Python库需要）
    build-essential libssl-dev libffi-dev python3.8-dev \
    && ln -s /usr/bin/python3.8 /usr/bin/python \  # 软链接python为3.8
    && ln -s /usr/bin/pip3 /usr/bin/pip \          # 软链接pip为pip3
    && apt-get clean && rm -rf /var/lib/apt/lists/*  # 清理缓存减小体积

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖（使用清华源加速，处理大依赖超时问题）
# 注意：依赖中CUDA 12.x库可能与基础镜像CUDA 11.8兼容（多数情况下向下兼容）
RUN pip install --upgrade pip \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 180 --no-cache-dir -r requirements.txt

# 复制所有代码文件（脚本、dataset等）
COPY . .

# 容器启动命令（替换为你的主脚本，如a_demonstration.py）
CMD ["python", "a_demonstration.py"]