from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model configuration
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Base model to fine-tune
    output_dir = "./output"  # Directory to save the fine-tuned model
    
    # Training parameters
    num_train_epochs = 3
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    max_grad_norm = 0.3
    weight_decay = 0.001
    
    # LoRA configurations
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    
    # Data processing
    max_seq_length = 512
    preprocessing_num_workers = 4
    
    # Evaluation and logging
    evaluation_strategy = "steps"
    eval_steps = 200
    save_strategy = "steps"
    save_steps = 200
    logging_steps = 10
    
    # Mixed precision training
    fp16 = True
    
    # Other
    seed = 42
    push_to_hub = False 