from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from peft import LoraConfig, get_peft_model

@dataclass
class ModelArguments:
    model_name: str
    max_length: int
    lora_config: Dict[str, Any]

class GemmaModel:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_args = ModelArguments(
            model_name=config['model']['name'],
            max_length=config['model']['max_length'],
            lora_config=config['model']['lora']
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name)
        self.model = self._setup_model()
        
    def _setup_model(self):
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.model_args.lora_config['r'],
            lora_alpha=self.model_args.lora_config['alpha'],
            lora_dropout=self.model_args.lora_config['dropout'],
            target_modules=self.model_args.lora_config['target_modules'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model