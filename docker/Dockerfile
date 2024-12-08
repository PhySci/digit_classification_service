FROM python:3.10

RUN mkdir -p /app
COPY src /app

WORKDIR /app

RUN pip install -r requirements.txt

COPY models /models

CMD ["python", "app.py"]
