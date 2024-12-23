
# -*- coding: utf-8 -*-
"""
@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

from pydub import AudioSegment
import numpy as np
import simpleaudio as sa
from NOTE_FREQUENCY import NOTE_FREQUENCIES


def read_text_file(file_path):
    """
    Reads the contents of a text file and returns it as a string.
    
    :param file_path: Path to the text file
    :return: Content of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except Exception as e:
        return f"An error occurred: {e}"

# Generate a sine wave for a given frequency
def generate_sine_wave(frequency, duration_ms, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000,
                    int(sample_rate * duration_ms / 1000), False)
    wave = 0.5 * amplitude * np.sin(2 * np.pi * frequency * t)
    wave = (wave * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        wave.tobytes(),
        frame_rate=sample_rate,
        sample_width=wave.dtype.itemsize,
        channels=1
    )
    return audio_segment

# Function to create a sequence of notes


def create_sequence(note_sequence, duration_ms=500):
    song = AudioSegment.silent(duration=0)
    for note in note_sequence:
        if note == 'R':  # Handle rest
            segment = AudioSegment.silent(duration=duration_ms)
        else:
            frequency = NOTE_FREQUENCIES[note]
            segment = generate_sine_wave(frequency, duration_ms)
        song += segment
    return song


def play_note_sequence(note_sequence, duration_ms=500):
    # Example sequence (You can replace this with your sequence)
    # sequence = "C C G G A A G F F E E D D C G G F F E E D G G F F E E D C C G G A A G F F E E D D C".split()
    # Famous tune sequence: "Twinkle Twinkle Little Star"
    # sequence = "C C G G A A G R F F E E D D C R G G F F E E D R C C G G A A G R F F E E D D C".split()

    # Create the sequence
    song = create_sequence(
        note_sequence, duration_ms=duration_ms)  # 500ms per note

    # Save the song to a .wav file
    song.export("output/nursery_rhyme.wav", format="wav")

    # Play the .wav file using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file("output/nursery_rhyme.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()


if __name__ == "__main__":
    print("Reading generated melody from file...")
    note: str = read_text_file("output/generated_melody.txt").split()
    play_note_sequence(note, 500)
