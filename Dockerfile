FROM python:3.12
WORKDIR /app
ADD app /app
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "main.py", "-u"]