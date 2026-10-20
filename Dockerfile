FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/IRHP
COPY . .
RUN python -m pip install --no-cache-dir -r requirements.txt
# NVIDIA Container Toolkit belongs on the host, not inside this container.
CMD ["python", "-m", "irhp.train", "--help"]
