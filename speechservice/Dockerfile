# 3.13, not core's 3.14: kokoro-onnx 0.6 does not support 3.14 yet.
FROM python:3.13-slim

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

WORKDIR /app
COPY . /app

# Model cache: k8s/compose mounts the speech volume here (HF_HOME).
ENV HF_HOME=/models
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
