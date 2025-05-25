FROM python:3.12
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
ADD app /app

ENTRYPOINT ["python3", "-u", "main.py"]