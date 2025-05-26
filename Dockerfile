FROM python:3.12
COPY requirements.txt .
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install -r requirements.txt

WORKDIR /app
ADD app /app

ENTRYPOINT ["python3", "-u", "main.py"]