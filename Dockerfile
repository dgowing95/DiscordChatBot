FROM python:3.12
COPY requirements.txt .
RUN pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip --no-cache-dir install -r requirements.txt

WORKDIR /app
ADD app /app

ENTRYPOINT ["python3", "-u", "main.py"]