FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[gemini,otel]"
ENV LLMSCOPE_DB=/data/llmscope.db PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn llmscope.api:app --host 0.0.0.0 --port ${PORT}"]
