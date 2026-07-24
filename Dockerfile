FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Defaults to the offline mock backend so `docker run` works with zero
# configuration. Override at runtime, e.g.:
#   docker run --env-file .env myimage python -m src.cli langgraph "..."
ENV LLM_PROVIDER=mock

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["langgraph", "What is 12 * (4 + 3)?"]
