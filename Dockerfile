# RTX 4090 (Ada Lovelace) 최적화를 위해
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

# 시스템 타임존 및 비대화형 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Ada Lovelace 아키텍처 타겟팅 (FlashAttention-2 최적화)
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV I_AM_A_DOCKER_CONTAINER=1

# OS 레벨 필수 패키지 설치 (DICOM 및 의료 영상 전처리용)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 컨테이너 실행 시 기본 쉘
CMD ["/bin/bash"]
