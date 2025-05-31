FROM python:3.12
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip --no-cache-dir install -r requirements.txt

WORKDIR /app
ADD . /app

ENTRYPOINT ["python", "-m", "uvicorn", "--host", "0.0.0.0", "main:app"]