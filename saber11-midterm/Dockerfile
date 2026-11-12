FROM python:3.11-slim-bookworm

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY predict.py model.bin ./

EXPOSE 9696

CMD ["uv", "run", "--no-dev", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
