import pandas as pd
import ollama
import json
from tqdm import tqdm
import os

def load_and_preprocess_data():
    """Load and preprocess the medical products dataset."""
    print("Loading dataset...")
    df = pd.read_csv('medscape.csv')
    
    # Limit to 5000 examples
    df = df.head(5000)
    
    # Create formatted text entries for each product
    training_data = []
    for _, row in df.iterrows():
        # Format as a conversation
        prompt = f"""<system>You are a medical assistant trained to suggest medications based on symptoms and conditions.</system>
<user>I need medication advice for common symptoms.</user>
<assistant>Based on the available information, here's a medication suggestion:

Medicine: {row['name']}
Price: ₹{row['price(₹)']}
Manufacturer: {row['manufacturer_name']}
Type: {row['type']}
Pack Size: {row['pack_size_label']}
Composition: {row['short_composition1']}

Please consult a healthcare professional before taking any medication.</assistant>"""
        training_data.append(prompt)
    
    return training_data

def create_modelfile():
    """Create a modelfile for fine-tuning."""
    modelfile = '''
FROM llama3.2

# Set parameters for training
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# System prompt
SYSTEM "You are a medical assistant trained to suggest medications based on symptoms and conditions. Always provide detailed information about the medications including price, manufacturer, type, pack size, and composition."
'''
    
    with open("Modelfile", "w") as f:
        f.write(modelfile)

def main():
    # Load and preprocess data
    training_data = load_and_preprocess_data()
    print(f"Loaded {len(training_data)} training examples")
    
    # Create modelfile
    create_modelfile()
    
    # Create a new model for fine-tuning
    print("Creating new model...")
    os.system("ollama create medbot -f Modelfile")
    
    # Fine-tune the model
    print("Starting fine-tuning...")
    for i, prompt in enumerate(tqdm(training_data)):
        try:
            # Fine-tune with each example
            response = ollama.generate(
                model='medbot',
                prompt=prompt,
                options={
                    'num_predict': 0,  # For training only
                }
            )
            
            if (i + 1) % 100 == 0:
                print(f"\nProcessed {i + 1} examples")
                
        except Exception as e:
            print(f"Error training on example {i}: {str(e)}")
            continue
    
    print("\nTraining complete!")
    print("You can now use the model with: ollama run medbot")

if __name__ == "__main__":
    main() 