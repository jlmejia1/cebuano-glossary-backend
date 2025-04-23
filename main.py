import fitz  # PyMuPDF for PDF text extraction
import os
import re
import concurrent.futures
import cebstemmer.stemmer as original_stemmer
from zipf_filter import apply_zipf_filter
from tfidf import compute_tf_idf  # Replace old TF-IDF imports
import subprocess  # <-- Added for glossary generation

# Cebuano stopwords list
stopwords = set([
    "ako", "kita", "amua", "ato", "atoa", "ikaw", "imong", "imo", "akoa", "akong kaugalingon",
    "kaugalingon", "kamo mismo", "siya", "iya", "kaniya", "mismo", "sila", "ila", "ilahang",
    "ilang", "nila", "unsa", "nga", "kinsa", "kini", "kana", "mga", "kadtong mga", "kaniadto",
    "niatong", "mahimong", "nahimo", "ikaw'ng", "ikaw ang", "kitang", "kinsa'ng", "kinsang",
    "kinsa ang", "unsa'ng", "unsa ang", "unsang", "kanus-ang", "kanusang", "kansang", "asang",
    "nganong", "ngano'ng", "giunsa'ng", "ginunsang", "ang", "ug", "pero", "kung", "o", "tungod",
    "sama", "hangtud", "hantud", "samtang", "sa", "pinaagi", "para", "uban sa", "mahitungod",
    "batok", "tali sa", "pinaagi sa", "sa panahon", "kaniadto", "human", "itaas", "ibabaw",
    "ubos", "gikan", "taas", "sulod", "gawas", "ibabaw", "ilawum", "napud", "unya", "kas-a",
    "diri", "didto", "kanus-a", "asa", "ngano", "giunsa", "tanan", "kada", "dyutay", "pipila",
    "mas", "kina", "uban", "uban pa", "sama", "dili", "lamang", "pareho", "busa", "kay", "pud",
    "kaayo"
])

# Fix stemmer function
def fixed_strip_prefix(stem):
    prefixes = original_stemmer.prefixes()
    longest_prefix = max((p for p in prefixes if stem.root.startswith(p)), key=len, default="")
    if longest_prefix:
        stem.root = stem.root.replace(longest_prefix, "", 1)
    return stem

original_stemmer.strip_prefix = fixed_strip_prefix


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize_text(text, min_length=3):
    tokens = text.split()
    return [word for word in tokens if len(word) >= min_length]


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]


def stem_cebuano_word(word):
    morphemes = original_stemmer.stem_word(word)
    return morphemes[0] if morphemes else word


def process_pdfs(pdf_folder,
                 output_folder="pdf-to-text",
                 clean_output_folder="cleaned",
                 tokenized_output_folder="tokenized",
                 filtered_output_folder="filtered",
                 stemmed_output_folder="stemmed",
                 validated_output_folder="validated",
                 zipf_filtered_folder=None,
                 zipf_csv_folder=None,
                 zipf_words_folder=None,
                 tfidf_scores_folder=None,
                 tfidf_words_folder=None):
    # Create all output directories (skip None)
    folders = [
        output_folder, clean_output_folder, tokenized_output_folder,
        filtered_output_folder, stemmed_output_folder, validated_output_folder,
        zipf_filtered_folder, zipf_csv_folder, zipf_words_folder,
        tfidf_scores_folder, tfidf_words_folder
    ]
    for folder in folders:
        if folder:
            os.makedirs(folder, exist_ok=True)

    # Process each PDF file
    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_folder, filename)
        text = ""

        # Extract text from PDF
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"

        base_name = os.path.splitext(filename)[0]
        txt_filename = base_name + ".txt"

        # Save raw extracted text
        with open(os.path.join(output_folder, txt_filename), "w", encoding="utf-8") as f:
            f.write(text)

        # Clean text
        cleaned_text = clean_text(text)
        with open(os.path.join(clean_output_folder, txt_filename), "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Tokenize
        tokenized_words = tokenize_text(cleaned_text)
        with open(os.path.join(tokenized_output_folder, txt_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(tokenized_words))

        # Remove stopwords
        filtered_words = remove_stopwords(tokenized_words)
        with open(os.path.join(filtered_output_folder, txt_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_words))

        # Stem words
        stemmed_words = [stem_cebuano_word(word) for word in filtered_words]
        with open(os.path.join(stemmed_output_folder, txt_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(stemmed_words))

        # Save validated words (input for filters)
        with open(os.path.join(validated_output_folder, txt_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(stemmed_words))

    # After stemming is complete, run pipelines if configured
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        if zipf_words_folder:
            executor.submit(
                apply_zipf_filter,
                input_folder=validated_output_folder,
                output_folder=zipf_filtered_folder,
                csv_output_folder=zipf_csv_folder,
                glossary_output_folder=zipf_words_folder
            )
        if tfidf_words_folder:
            executor.submit(
                compute_tf_idf,
                input_folder=validated_output_folder,
                output_scores_folder=tfidf_scores_folder,
                output_words_folder=tfidf_words_folder
            )

    print("\n✅ Processing complete!")
    print(f"Original texts: {output_folder}")
    print(f"Cleaned texts: {clean_output_folder}")
    print(f"Tokenized words: {tokenized_output_folder}")
    print(f"Filtered words: {filtered_output_folder}")
    print(f"Stemmed words: {stemmed_output_folder}")
    if zipf_words_folder:
        print(f"Zipf results: {zipf_words_folder}")
    if tfidf_words_folder:
        print(f"TF-IDF results: {tfidf_words_folder}")
    if tfidf_scores_folder and tfidf_words_folder:
        print("TF-IDF threshold calculated automatically (mean + 1SD)")

    # ---- Added for glossary generation ----
    glossary_output_folder = os.path.join(pdf_folder, "glossary-csv")
    html_file = "pg40074-images.html"
    subprocess.run([
        "python", "generate_glossary.py",
        zipf_words_folder or "zipf-words",
        tfidf_words_folder or "tfidf-words",
        html_file,
        glossary_output_folder
    ])
    print(f"Glossary CSV files saved in: {glossary_output_folder}")
    # ---- End glossary addition ----


if __name__ == "__main__":
    # Define all paths
    pdf_folder = r"C:/Users/bluep/Documents/CMSC-198/Dataset(PDF)"

    # Main output folders
    pdf_text_folder = os.path.join(pdf_folder, "pdf-to-text")
    cleaned_text_folder = os.path.join(pdf_folder, "cleaned")
    tokenized_text_folder = os.path.join(pdf_folder, "tokenized")
    filtered_text_folder = os.path.join(pdf_folder, "filtered")
    stemmed_text_folder = os.path.join(pdf_folder, "stemmed")
    validated_text_folder = os.path.join(pdf_folder, "validated")

    # Zipf outputs
    zipf_filtered_folder = os.path.join(pdf_folder, "zipf-filtered")
    zipf_csv_folder = os.path.join(pdf_folder, "zipf-csv-scores")
    zipf_words_folder = os.path.join(pdf_folder, "zipf-words")

    # TF-IDF outputs
    tfidf_scores_folder = os.path.join(pdf_folder, "tfidf-scores")
    tfidf_words_folder = os.path.join(pdf_folder, "tfidf-words")

    # Run the complete pipeline
    process_pdfs(
        pdf_folder=pdf_folder,
        output_folder=pdf_text_folder,
        clean_output_folder=cleaned_text_folder,
        tokenized_output_folder=tokenized_text_folder,
        filtered_output_folder=filtered_text_folder,
        stemmed_output_folder=stemmed_text_folder,
        validated_output_folder=validated_text_folder,
        zipf_filtered_folder=zipf_filtered_folder,
        zipf_csv_folder=zipf_csv_folder,
        zipf_words_folder=zipf_words_folder,
        tfidf_scores_folder=tfidf_scores_folder,
        tfidf_words_folder=tfidf_words_folder
    )
