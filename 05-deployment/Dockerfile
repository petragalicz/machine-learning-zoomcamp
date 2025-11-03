FROM agrigorev/zoomcamp-model:2025
RUN pip install uv

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --locked

COPY "predict.py" "pipeline_v1.bin" ./

RUN pip install --no-cache-dir scikit-learn fastapi uvicorn

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

