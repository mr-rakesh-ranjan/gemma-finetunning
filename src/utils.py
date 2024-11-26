import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_project_structure():
    directories = [
        'config',
        'data/raw',
        'data/processed',
        'src',
        'scripts',
        'notebooks',
        'outputs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)