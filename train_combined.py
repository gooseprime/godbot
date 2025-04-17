import pandas as pd
import ollama
import json
from tqdm import tqdm
import os
import csv

def load_period_data():
    """Load and preprocess the period dataset."""
    print("Loading period dataset...")
    df = pd.read_csv('period - Copy.csv')
    
    # Create formatted text entries for each record
    training_data = []
    for _, row in df.iterrows():
        # Format as a conversation
        prompt = f"""<system>You are a medical assistant trained to provide information about menstrual health and mental health.</system>
<user>I need information about menstrual cycles and related health.</user>
<assistant>Based on the available information, here's what I can tell you about menstrual cycles:

Age: {row['Age']}
Length of cycle: {row['Length_of_cycle']} days
Estimated day of ovulation: {row['Estimated_day_of_ovulution']}
Length of Luteal Phase: {row['Length_of_Leutal_Phase']} days
Length of menses: {row['Length_of_menses']} days
Unusual Bleeding: {row['Unusual_Bleeding']}
Height: {row['Height']}
Weight: {row['Weight']} kg
BMI: {row['BMI']}
Mean length of cycle: {row['Mean_of_length_of_cycle']} days
Menses score: {row['Menses_score']}

This information can help you understand your menstrual cycle patterns. Please consult a healthcare professional for personalized advice.</assistant>"""
        training_data.append(prompt)
    
    return training_data

def load_dass_data():
    """Load and preprocess the DASS dataset."""
    print("Loading DASS dataset...")
    try:
        # Try reading with default settings first
        df = pd.read_csv('data.csv')
    except pd.errors.ParserError:
        print("Initial parsing failed, trying with more robust settings...")
        # If that fails, try with more robust settings
        df = pd.read_csv('data.csv', 
                        quoting=csv.QUOTE_ALL,  # Quote all fields
                        escapechar='\\',        # Use backslash as escape character
                        on_bad_lines='skip')    # Skip problematic lines
    
    # Randomly sample 10,000 rows if we have more than that
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)  # Using random_state for reproducibility
    
    training_data = []
    
    # Process each row in the dataset
    for _, row in df.iterrows():
        # Format as a conversation
        prompt = f"""<s>You are EDI (Emotional Diagnostic Intelligence), a medical assistant trained to provide mental health assessments.</s>
<user>I need an analysis of my DASS responses.</user>
<assistant>I'll analyze your DASS (Depression Anxiety Stress Scales) responses:

"""
        
        # Add DASS questions and responses
        for col in df.columns:
            if col.startswith('Q'):  # Assuming DASS questions start with 'Q'
                value = row[col]
                if pd.notna(value):  # Only include non-null values
                    prompt += f"{col}: {value}\n"
        
        # Add personality traits if available
        tipi_cols = [col for col in df.columns if 'TIPI' in col]
        if tipi_cols:
            prompt += "\nPersonality Traits:\n"
            for col in tipi_cols:
                value = row[col]
                if pd.notna(value):
                    prompt += f"{col}: {value}\n"
        
        prompt += "\nBased on these responses, I can provide an assessment of your mental health status. Would you like me to analyze specific aspects or provide an overall evaluation?</assistant>"
        
        training_data.append(prompt)
    
    return training_data

def create_modelfile():
    """Create a modelfile for fine-tuning."""
    modelfile = '''
FROM medbot

# Set parameters for training
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# System prompt
SYSTEM """You are EDI (Emotional Diagnostic Intelligence), a medical assistant created by Illusiveman, trained to provide information about medications, menstrual health, and mental health. Always provide detailed information and recommend consulting healthcare professionals for personalized advice."""
'''
    
    with open("Modelfile_combined", "w") as f:
        f.write(modelfile)

def main():
    # Load and preprocess data from both datasets
    period_data = load_period_data()
    dass_data = load_dass_data()
    
    # Combine the datasets
    training_data = period_data + dass_data
    
    print(f"Loaded {len(period_data)} period examples and {len(dass_data)} DASS examples")
    print(f"Total training examples: {len(training_data)}")
    
    # Create modelfile
    create_modelfile()
    
    # Create a new model for fine-tuning
    print("Creating new model...")
    os.system("ollama create edi -f Modelfile_combined")
    
    # Fine-tune the model
    print("Starting fine-tuning...")
    for i, prompt in enumerate(tqdm(training_data)):
        try:
            # Fine-tune with each example
            response = ollama.generate(
                model='edi',
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
    print("You can now use the model with: ollama run edi")

if __name__ == "__main__":
    main() 