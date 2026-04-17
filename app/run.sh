#/bin/bash

uv run -m src/models/train.py
uv run uvicorn app.api:app --host "0.0.0.0"