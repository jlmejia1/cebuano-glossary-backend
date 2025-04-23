import os
import pandas as pd
import math
from collections import Counter
from cebdict import dictionary


def get_count_limits(corpus_size):
    """
    Calculate stricter frequency limits using:
    - 4th root scaling (N^(1/4)) for max count
    - Higher minimum Zipf (0.0002)
    - Lower maximum Zipf (0.0008)
    """
    return {
        'max_count': min(30, int(corpus_size ** (1 / 4))),  # Much stricter than N^(1/3)
        'min_zipf': 0.0002,  # ~5 occ/25k words
        'max_zipf': 0.0008  # ~20 occ/25k words
    }


def is_valid_glossary_word(word, global_count, global_zipf, corpus_size):
    """
    Stricter validation with:
    - Higher minimum count (4)
    - Tighter Zipf range
    - Preserved dictionary/format checks
    """
    limits = get_count_limits(corpus_size)
    return (
            (4 <= global_count <= limits['max_count']) and  # Min count raised to 4
            (limits['min_zipf'] <= global_zipf <= limits['max_zipf']) and
            dictionary.is_entry(word) and
            word.isalpha()
    )


def calculate_zipf_per_file(input_folder, output_csv_folder, output_glossary_folder):
    """Generates glossaries with stricter thresholds"""
    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_glossary_folder, exist_ok=True)

    # Build global corpus
    global_corpus = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as f:
                global_corpus.extend(f.read().splitlines())

    if not global_corpus:
        print("⚠️ No words found. Check input files.")
        return

    global_word_freq = Counter(global_corpus)
    corpus_size = len(global_corpus)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as f:
            words = f.read().splitlines()

        if not words:
            continue

        # Process words
        word_data = []
        glossary_words = set()
        local_word_freq = Counter(words)

        for word, local_count in local_word_freq.items():
            global_count = global_word_freq.get(word, 0)
            global_zipf = global_count / corpus_size

            word_data.append({
                'word': word,
                'local_count': local_count,
                'global_count': global_count,
                'global_zipf': global_zipf,
                'is_valid': dictionary.is_entry(word)
            })

            if is_valid_glossary_word(word, global_count, global_zipf, corpus_size):
                glossary_words.add(word)

        # Save outputs
        df = pd.DataFrame(word_data)
        csv_path = os.path.join(output_csv_folder, f"{os.path.splitext(filename)[0]}_zipf.csv")
        df.to_csv(csv_path, index=False)

        if glossary_words:
            glossary_path = os.path.join(output_glossary_folder, f"{os.path.splitext(filename)[0]}_glossary.txt")
            with open(glossary_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(sorted(glossary_words)))


def apply_zipf_filter(input_folder, output_folder, csv_output_folder, glossary_output_folder):
    """Full pipeline with stricter thresholds"""
    calculate_zipf_per_file(input_folder, csv_output_folder, glossary_output_folder)
    print(f"Glossaries saved to: {glossary_output_folder}")