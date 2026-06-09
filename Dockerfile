# RTX 4090 (Ada Lovelace) 최적화를 위해
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

# 시스템 타임존 및 비대화형 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Ada Lovelace 아키텍처 타겟팅 (FlashAttention-2 최적화)
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV I_AM_A_DOCKER_CONTAINER=1

# 한국 미러로 전환 (archive.ubuntu.com 이 본 네트워크에서 차단됨)
RUN sed -i 's|http://archive.ubuntu.com|http://kr.archive.ubuntu.com|g; s|http://security.ubuntu.com|http://kr.archive.ubuntu.com|g' \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true

# OS 레벨 필수 패키지 설치 (DICOM 및 의료 영상 전처리용)
# Ubuntu 24.04(noble) 부터 libgl1-mesa-glx → libgl1 로 패키지명 변경됨
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# 파이썬 패키지 설치
# Ubuntu 24.04 의 PEP 668 (externally-managed-environment) 우회 - 컨테이너 안에서는 안전
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --upgrade pip && \
    pip install --no-cache-dir --break-system-packages -r requirements.txt

# 컨테이너 실행 시 기본 쉘
CMD ["/bin/bash"]
