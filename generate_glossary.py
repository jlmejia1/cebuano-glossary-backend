import os
import json
import re
import unicodedata


def remove_diacritics(text):
    """Converts accented characters to their base form (e.g., 'bítaw' → 'bitaw')."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def generate_glossary(zipf_folder, tfidf_folder, html_file, output_folder):
    """Creates a JSON glossary for each file by merging words from Zipf and TF-IDF outputs."""
    os.makedirs(output_folder, exist_ok=True)

    # Process each corresponding file in the zipf folder
    for filename in os.listdir(zipf_folder):
        if not filename.endswith(".txt"):
            continue

        base_name = os.path.splitext(filename)[0]
        zipf_file = os.path.join(zipf_folder, filename)
        tfidf_file = os.path.join(tfidf_folder, filename)

        # Read words from the Zipf file
        with open(zipf_file, "r", encoding="utf-8") as f:
            zipf_words = {line.strip() for line in f if line.strip()}

        # Read words from the TF-IDF file if it exists
        tfidf_words = set()
        if os.path.exists(tfidf_file):
            with open(tfidf_file, "r", encoding="utf-8") as f:
                tfidf_words = {line.strip() for line in f if line.strip()}

        # Merge words for this file and sort alphabetically (case-insensitive)
        merged_words = sorted(zipf_words.union(tfidf_words), key=lambda w: w.lower())

        # Save merged words as JSON list
        output_file = os.path.join(output_folder, base_name + ".json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_words, f, ensure_ascii=False, indent=2)

    print(f"✅ Glossary JSON files saved in: {output_folder}")
