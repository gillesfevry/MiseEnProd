#/bin/bash

uv run src/models/train.py
uv run uvicorn app.api:app --host "0.0.0.0"