# Initialize Unsloth with Ollama model
print("Initializing model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama3.2",  # Using the local Ollama Llama 3.2 model
    max_seq_length=512,
    dtype=None,  # Will be automatically determined
    load_in_4bit=True,  # Use 4-bit quantization
) 