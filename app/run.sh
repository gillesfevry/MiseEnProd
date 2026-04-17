#/bin/bash

uv run -m src/models/train
uv run uvicorn app.api:app --host "0.0.0.0"