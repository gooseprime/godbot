# Llama 2 Fine-tuning with Medscape Dataset

This project fine-tunes the Llama 2 model using the Medscape medical dataset using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the `medscape.csv` file in the root directory with 'question' and 'answer' columns.

3. (Optional) Adjust the training parameters in `config.py` according to your needs and hardware capabilities.

## Training

To start the fine-tuning process, run:
```bash
python train.py
```

The script will:
1. Load and preprocess the Medscape dataset
2. Initialize the Llama 2 model with LoRA configuration
3. Train the model using the specified parameters
4. Save the fine-tuned model in the `output` directory

## Configuration

The training configuration can be modified in `config.py`. Key parameters include:
- Model configuration (base model, output directory)
- Training parameters (epochs, batch sizes, learning rate)
- LoRA configurations (rank, alpha, dropout)
- Data processing parameters
- Evaluation and logging settings

## Hardware Requirements

- GPU with at least 16GB VRAM recommended
- At least 32GB system RAM recommended
- Sufficient disk space for model checkpoints

## Notes

- The training uses mixed precision (fp16) by default for better memory efficiency
- The model is fine-tuned using LoRA for parameter-efficient training
- Training progress can be monitored through the console output
- Checkpoints are saved according to the specified save steps

## Output

The fine-tuned model will be saved in the `output` directory. Each checkpoint will include:
- Model weights
- Training configuration
- Optimizer states
- Training logs # godbot
