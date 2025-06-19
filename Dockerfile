FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 🛠️ System dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    python3-pip \
    python3-dev \
    libsndfile1 \
    mpg123 \
    && rm -rf /var/lib/apt/lists/*

# 🔗 Fix python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# 📦 Upgrade pip
RUN pip install --upgrade pip

# 🔥 Install PyTorch with CUDA 12.1
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# 📚 Install Python dependencies (except bitsandbytes)
RUN pip install \
    transformers \
    langchain \
    langchain-community \
    openai-whisper \
    ffmpeg-python \
    gTTS \
    scipy \
    sentencepiece \
    protobuf \
    accelerate \
    gradio \
    playsound==1.2.2 

# 🧠 BitsAndBytes
RUN pip install --no-cache-dir bitsandbytes==0.46.0

# 📂 Set work directory
WORKDIR /app

# 📁 Copy app code
COPY . .

# 🚀 Default run command
CMD ["python", "app_per_instruct_model.py"]
