import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from config import TrainingConfig

def load_and_preprocess_data():
    """Load and preprocess the medical products dataset."""
    df = pd.read_csv('medscape.csv')
    
    # Create formatted text entries for each product
    def format_product_info(row):
        info = f"### Product: {row['name']}\n\n"
        info += f"Description:\n"
        info += f"- Price: ₹{row['price(₹)']}\n"
        info += f"- Manufacturer: {row['manufacturer_name']}\n"
        info += f"- Type: {row['type']}\n"
        info += f"- Pack Size: {row['pack_size_label']}\n"
        info += f"- Composition: {row['short_composition1']}"
        info += "\n\n"
        return info
    
    df['text'] = df.apply(format_product_info, axis=1)
    dataset = Dataset.from_pandas(df[['text']])
    return dataset

def prepare_model_and_tokenizer(config):
    """Prepare the model and tokenizer for training."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
        device_map="auto" if device.type == "mps" else None,
        trust_remote_code=True,
    )
    
    if device.type != "mps":
        model = model.to(device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the text data."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

def main():
    # Load configuration
    config = TrainingConfig()
    
    # Load and preprocess data
    dataset = load_and_preprocess_data()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        remove_columns=dataset.column_names,
        num_proc=config.preprocessing_num_workers,
    )
    
    # Split dataset
    train_val = tokenized_dataset.train_test_split(test_size=0.1, seed=config.seed)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        fp16=torch.backends.mps.is_available(),  # Use fp16 only if MPS is available
        push_to_hub=config.push_to_hub,
        seed=config.seed,
        use_mps_device=torch.backends.mps.is_available(),  # Enable MPS if available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_val["train"],
        eval_dataset=train_val["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
if __name__ == "__main__":
    main() 