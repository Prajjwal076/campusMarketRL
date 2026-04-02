FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY . /app

RUN pip install --no-cache-dir -r campus_market_env/server/requirements.txt \
    && pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "campus_market_env.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
