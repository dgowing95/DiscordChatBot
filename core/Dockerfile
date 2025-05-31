FROM python:3.12

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

WORKDIR /app
ADD . /app

ENTRYPOINT ["python3", "-W ignore", "-u", "main.py"]