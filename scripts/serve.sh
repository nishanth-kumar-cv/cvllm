#!/bin/bash
cd python_finetune
uvicorn infer_api:app --host 0.0.0.0 --port 8000
