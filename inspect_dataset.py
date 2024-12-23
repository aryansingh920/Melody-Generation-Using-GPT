def inspect_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    valid_tokens = set(['C', 'C#', 'D', 'D#', 'E', 'F',
                       'F#', 'G', 'G#', 'A', 'A#', 'B', 'R'])
    inconsistent_lines = []

    # Analyze each line
    for idx, line in enumerate(data):
        line = line.strip()  # Remove whitespace
        if not set(line).issubset(valid_tokens):
            inconsistent_lines.append((idx, line))

    # Output summary
    print(f"Total Lines: {len(data)}")
    print(f"Inconsistent Lines: {len(inconsistent_lines)}")
    if inconsistent_lines:
        print("\nSample of Inconsistent Lines:")
        # Show up to 5 problematic lines
        for idx, line in inconsistent_lines[:5]:
            print(f"Line {idx + 1}: {line}")
    else:
        print("All lines are consistent with the expected token structure.")


# Path to the output dataset
inspect_dataset("data/inputMelodiesAugmented.txt")
