#!/bin/bash

# Activate virtual environment (if using)
# source venv/bin/activate

# Process data
python -m src.data_processing \
    --config config/config.yaml \
    --input data/raw/ICD_TOP250_code.csv \
    --output data/processed/training_data.jsonl

# Run training
python -m src.trainer \
    --config config/config.yaml \
    --data data/processed/training_data.jsonl \
    --output outputs