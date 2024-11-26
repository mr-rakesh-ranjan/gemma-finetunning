# Gemma 2.0 Fine-tuning for ICD-10 Codes

This project implements fine-tuning of the Gemma 2.0 model for ICD-10 code classification and description tasks.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your settings in `config/config.yaml`

4. Place your raw data in `data/raw/`

## Training

1. Process the data:
```bash
python -m src.data_processing
```

2. Start training:
```bash
bash scripts/train.sh
```

## Project Structure

- `config/`: Configuration files
- `data/`: Data directory
  - `raw/`: Raw input data
  - `processed/`: Processed data ready for training
- `src/`: Source code
  - `data_processing.py`: Data preprocessing
  - `model.py`: Model configuration
  - `trainer.py`: Training logic
  - `utils.py`: Utility functions
- `scripts/`: Training scripts
- `notebooks/`: Jupyter notebooks for exploration
- `outputs/`: Trained models and checkpoints

## Best Practices

1. Always version your data and model checkpoints
2. Use wandb for experiment tracking
3. Use LoRA for efficient fine-tuning
4. Implement proper validation splits
5. Monitor training metrics and validate results
6. Use fp16 training for efficiency
7. Implement proper error handling and logging

## License

MIT License