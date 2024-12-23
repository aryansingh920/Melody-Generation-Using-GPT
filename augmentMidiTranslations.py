# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:06:10 2024

@author: Giovanni Di Liberto
See description in the assignment instructions.
"""
from NOTE_FREQUENCY import NOTE_FREQUENCIES

# List of notes in order
NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']

def translate_notes(notes, shift):
    translated_notes = []
    for note in notes:
        if note in NOTES:
            index = NOTES.index(note)
            new_index = (index + shift) % len(NOTES)
            translated_notes.append(NOTES[new_index])
        else:
            translated_notes.append(note)  # Keep the character as is if it's not a note
    return ''.join(translated_notes)

# Example usage
#input_notes = "CDE"
#shift = 1
#output_notes = translate_notes(input_notes, shift)
#print(output_notes)  # Output: cdF


# Load the input file
with open('data/inputMelodies.txt', 'r') as file:
    input_melodies = file.readlines()

# Apply 5 different translations and save the results
shifts = [1, 2, 3, 4, 5]
augmented_melodies = []

for shift in shifts:
    for melody in input_melodies:
        augmented_melodies.append(translate_notes(melody.strip(), shift))

# Save the augmented melodies to a new file
with open('data/inputMelodiesAugmented.txt', 'w') as file:
    for melody in augmented_melodies:
        file.write(melody + '\n')

print("The augmented melodies have been saved to inputMelodiesAugmented.txt")


