import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from cebdict import dictionary  # For Cebuano word validation


def calculate_dynamic_threshold(tfidf_scores):
    """
    Calculate automatic threshold using mean + 1SD of valid Cebuano words.
    Only considers words that are valid Cebuano entries in the dictionary.
    """
    if not tfidf_scores:
        return 0.01  # Fallback threshold

    mean = np.mean(tfidf_scores)
    std = np.std(tfidf_scores)
    return mean + std


def is_valid_cebuano_word(word):
    """Check if a word is a valid Cebuano word using cebdict"""
    return (dictionary.is_entry(word) and
            word.isalpha() and
            len(word) > 2)  # Minimum length of 3 characters


def compute_tf_idf(input_folder, output_scores_folder, output_words_folder):
    """
    Enhanced TF-IDF computation with:
    1. Automatic thresholding
    2. Strict Cebuano word validation
    3. Proper noun filtering
    """
    os.makedirs(output_scores_folder, exist_ok=True)
    os.makedirs(output_words_folder, exist_ok=True)

    # First pass: Document frequency calculation
    doc_freq = defaultdict(int)
    documents = []
    valid_words_tracker = set()  # Track all valid Cebuano words found

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as f:
                words = [word.strip() for word in f.read().splitlines()]
                documents.append((filename, words))

                # Only count document frequency for valid Cebuano words
                unique_words = set(words)
                for word in unique_words:
                    if is_valid_cebuano_word(word):
                        doc_freq[word] += 1
                        valid_words_tracker.add(word)

    total_docs = len(documents)
    all_valid_tfidf_scores = []  # For threshold calculation (valid words only)

    # Second pass: Score computation
    for filename, words in documents:
        term_freq = defaultdict(int)
        total_terms = len(words)

        # Calculate term frequencies (including all words)
        for word in words:
            term_freq[word] += 1

        # Compute TF-IDF scores
        results = []

        for word, tf in term_freq.items():
            tf_score = tf / total_terms
            idf_score = math.log((total_docs) / (doc_freq.get(word, 0) + 1)) + 1
            tfidf = tf_score * idf_score
            is_valid = is_valid_cebuano_word(word)

            results.append({
                'word': word,
                'tf': tf_score,
                'idf': idf_score,
                'tfidf': tfidf,
                'is_valid': is_valid
            })

            if is_valid:
                all_valid_tfidf_scores.append(tfidf)

        # Save scores CSV
        df = pd.DataFrame(results).sort_values('tfidf', ascending=False)
        score_filename = os.path.splitext(filename)[0] + '_tfidf.csv'
        df.to_csv(os.path.join(output_scores_folder, score_filename), index=False)

    # Calculate threshold using only valid Cebuano words
    threshold = calculate_dynamic_threshold(all_valid_tfidf_scores)
    print(f"📊 Automatic TF-IDF threshold: {threshold:.4f} (mean + 1SD)")
    print(f"📝 Total valid Cebuano words found: {len(valid_words_tracker)}")

    # Third pass: Filter and save qualifying words
    for filename, words in documents:
        score_file = os.path.join(output_scores_folder, os.path.splitext(filename)[0] + '_tfidf.csv')
        df = pd.read_csv(score_file)

        # Filter words that meet all criteria:
        # 1. TF-IDF score >= threshold
        # 2. Valid Cebuano word
        # 3. Not a proper noun (first letter lowercase)
        qualifying_words = [
            word for word in df[
                (df['tfidf'] >= threshold) &
                (df['is_valid'] == True)
                ]['word'].tolist()
            if word[0].islower()  # Exclude proper nouns
        ]

        if qualifying_words:
            # Remove duplicates and sort alphabetically
            unique_words = sorted(set(qualifying_words))
            words_filename = os.path.splitext(filename)[0] + '_words.txt'

            with open(os.path.join(output_words_folder, words_filename), 'w', encoding='utf-8') as f:
                f.write('\n'.join(unique_words))

            print(f"✅ Saved {len(unique_words)} valid Cebuano words for {filename}")
        else:
            print(f"⚠️ No qualifying words found for {filename}")


if __name__ == "__main__":
    # Example usage (normally called from main.py)
    compute_tf_idf(
        input_folder="validated_texts",
        output_scores_folder="tfidf_scores",
        output_words_folder="tfidf_words"
    )