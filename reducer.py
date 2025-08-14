import pandas as pd
import os

# --- Configuration ---
input_folder = 'data'
output_folder = 'data'
input_filename = 'transactions_sample.csv'
output_filename = 'transactions_mini.csv'
sample_fraction = 0.45 # Takes 45% of the rows to be safely under 25MB

# --- Script Logic ---
input_path = os.path.join(input_folder, input_filename)
output_path = os.path.join(output_folder, output_filename)

try:
    print(f"Loading '{input_path}'...")
    df = pd.read_csv(input_path)
    
    print(f"Original number of rows: {len(df)}")
    
    # Take a random sample
    df_mini = df.sample(frac=sample_fraction, random_state=42)
    
    print(f"New number of rows: {len(df_mini)}")
    
    # Save the new, smaller file
    df_mini.to_csv(output_path, index=False)
    
    print(f"\nSuccess! A new, smaller file has been created at: '{output_path}'")
    
except FileNotFoundError:
    print(f"Error: Make sure the file '{input_path}' exists in your project folder.")