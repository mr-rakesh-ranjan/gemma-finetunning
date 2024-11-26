import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import yaml

class DataProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def create_training_examples(self, data: pd.DataFrame) -> List[Dict[str, str]]:
        training_examples = []
        
        for _, row in data.iterrows():
            # Basic code to description mapping
            example1 = {
                "input": f"What is the medical condition associated with ICD-10 code {row['CODE']}?",
                "output": f"The ICD-10 code {row['CODE']} represents {row['LONG DESCRIPTION (VALID ICD-10 FY2025)']}"
            }
            
            # Description to code mapping
            example2 = {
                "input": f"What is the ICD-10 code for {row['SHORT DESCRIPTION (VALID ICD-10 FY2025)']}?",
                "output": f"The ICD-10 code for {row['SHORT DESCRIPTION (VALID ICD-10 FY2025)']} is {row['CODE']}"
            }
            
            # Medical coding scenario
            example3 = {
                "input": f"A patient has been diagnosed with {row['LONG DESCRIPTION (VALID ICD-10 FY2025)']}. What is the appropriate ICD-10 code?",
                "output": f"For a patient diagnosed with {row['LONG DESCRIPTION (VALID ICD-10 FY2025)']}, the appropriate ICD-10 code is {row['CODE']}"
            }
            
            training_examples.extend([example1, example2, example3])
            
        return training_examples

    def process_data(self, input_path: str, output_path: str):
        df = pd.read_csv(input_path)
        examples = self.create_training_examples(df)
        
        # Save to JSONL format
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')