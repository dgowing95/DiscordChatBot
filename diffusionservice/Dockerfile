FROM python:3.12-slim

COPY requirements.txt .
# CUDA torch wheels (they bundle the nvidia-* runtime libs); cu128 works with
# CUDA >= 12.8 node drivers.
RUN pip --no-cache-dir install torch --index-url https://download.pytorch.org/whl/cu128
RUN pip --no-cache-dir install -r requirements.txt

WORKDIR /app
ADD . /app

# Model cache: k8s/compose mounts the diffusers volume here (HF_HOME).
ENV HF_HOME=/models
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
