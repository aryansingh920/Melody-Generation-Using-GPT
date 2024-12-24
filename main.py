"""
Created on 21/12/2024

@author: Aryan

Filename: main.py

Relative Path: main.py
"""

import os
from train_with_viz import test_configs_with_tracking
from evaluation import main as evaluate_models
# from gpt import generate_melody
# from melodyPlay import play_note_sequence


def main():
    """
    1) Trains multiple configurations of the GPT model using train_with_viz.py
    2) Evaluates all saved models on specified test files using evaluation.py
    3) (Optional) Generates melodies or plays them.
    """

    # Step 1: Train and test multiple configurations; returns the best model.
    print("Training model with multiple configurations...")
    best_model = test_configs_with_tracking()

    # Step 2: Evaluate all saved models on multiple test sets
    print("\nEvaluating saved models on the test sets...")
    evaluation_results = evaluate_models()
    print("Evaluation completed. Results have been saved to 'evaluation_results.json'.")

    # ----------------------------------------------------------------------------
    # Uncomment the lines below if you want to generate or play melodies directly:
    #
    # Step 3: Generate a melody from the best model
    # prompt = "R F G A "  # example prompt
    # generated_melody = generate_melody(prompt, max_new_tokens=100)
    # print("\nGenerated Melody:\n", generated_melody)
    #
    # Step 4: Play the generated melody (requires pydub & simpleaudio)
    # play_note_sequence(generated_melody.split(), duration_ms=500)
    #
    # ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
