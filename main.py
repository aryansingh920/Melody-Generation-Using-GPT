"""
Created on 21/12/2024

@author: Aryan

Filename: main.py

Relative Path: main.py
"""
from gpt import GPTLanguageModel, generate_melody
from train_with_viz import test_configs_with_tracking
# from evaluation import main as evaluate_model
# from melodyPlay import play_melody
import os

# Step 1: Load Dataset
data_path = "data/inputMelodiesAugmented.txt"
with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read().splitlines()

# Step 2: Train and Test Configurations
print("Training model with multiple configurations...")
best_model = test_configs_with_tracking()

# # Step 3: Evaluate Best Model
# evaluation_metrics = evaluate_model(best_model, data)
# print("Evaluation Metrics:", evaluation_metrics)

# # Step 4: Generate Melody
# prompt = "R F G A "  # Example starting prompt for melody generation
# generated_sequence = generate_melody(prompt, max_new_tokens=100)
# print("Generated Melody:", generated_sequence)

# # Step 5: Play Generated Melody
# output_file = "output/generated_melody.mid"
# play_melody(generated_sequence, output_file)
# print(f"Generated melody saved and played from {output_file}")
