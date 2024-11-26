import os
from typing import Dict, Any
import yaml
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

class GemmaTrainer:
    def __init__(self, model, tokenizer, config_path: str):
        self.model = model
        self.tokenizer = tokenizer
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def _load_dataset(self, data_path: str):
        dataset = load_dataset('json', data_files=data_path)
        
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples['input'],
                max_length=self.config['model']['max_length'],
                padding="max_length",
                truncation=True
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['output'],
                    max_length=self.config['model']['max_length'],
                    padding="max_length",
                    truncation=True
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
        
    def train(self, train_data_path: str):
        # Initialize wandb
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity']
        )
        
        # Load and process dataset
        dataset = self._load_dataset(train_data_path)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['model']['num_epochs'],
            per_device_train_batch_size=self.config['model']['batch_size'],
            per_device_eval_batch_size=self.config['model']['batch_size'],
            gradient_accumulation_steps=self.config['model']['gradient_accumulation_steps'],
            learning_rate=self.config['model']['learning_rate'],
            weight_decay=self.config['model']['weight_decay'],
            warmup_steps=self.config['model']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_steps=self.config['training']['save_steps'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            fp16=self.config['training']['fp16'],
            optim=self.config['training']['optim'],
            report_to="wandb"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True
            )
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        wandb.finish()