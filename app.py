from fastapi import FastAPI, UploadFile, File
import shutil, uuid, os, json
import firebase_admin
from firebase_admin import credentials, firestore
from main import process_pdfs
from generate_glossary import generate_glossary
from llama_client import define_word
from zipf_filter import apply_zipf_filter
from tfidf import compute_tf_idf

# Path to precomputed validated Cebuano corpus (relative to this file)
BASE_DIR = os.path.dirname(__file__)
GLOBAL_VALIDATED_FOLDER = os.path.join(BASE_DIR, "validated")

# Initialize Firebase Admin
cred = credentials.Certificate("service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    run_id = str(uuid.uuid4())
    tmp_dir = os.path.join("/tmp", run_id)
    os.makedirs(tmp_dir, exist_ok=True)

    # Save uploaded file
    pdf_path = os.path.join(tmp_dir, file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 1: Validate PDF only (no zipf/tfidf here)
    validated_new = os.path.join(tmp_dir, "validated")
    process_pdfs(
        pdf_folder=tmp_dir,
        output_folder=os.path.join(tmp_dir, "txt"),
        clean_output_folder=os.path.join(tmp_dir, "cleaned"),
        tokenized_output_folder=os.path.join(tmp_dir, "tokenized"),
        filtered_output_folder=os.path.join(tmp_dir, "filtered"),
        stemmed_output_folder=os.path.join(tmp_dir, "stemmed"),
        validated_output_folder=validated_new,
        zipf_filtered_folder=None,
        zipf_csv_folder=None,
        zipf_words_folder=None,
        tfidf_scores_folder=None,
        tfidf_words_folder=None
    )

    # Step 2: Combine validated files into a temporary corpus
    combined_dir = os.path.join(tmp_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    for fname in os.listdir(GLOBAL_VALIDATED_FOLDER):
        shutil.copy(os.path.join(GLOBAL_VALIDATED_FOLDER, fname), os.path.join(combined_dir, fname))
    for fname in os.listdir(validated_new):
        shutil.copy(os.path.join(validated_new, fname), os.path.join(combined_dir, fname))

    # Step 3: Run Zipf & TF-IDF on combined corpus
    zipf_dir = os.path.join(tmp_dir, "zipf-words")
    tfidf_dir = os.path.join(tmp_dir, "tfidf-words")
    os.makedirs(zipf_dir, exist_ok=True)
    os.makedirs(tfidf_dir, exist_ok=True)
    apply_zipf_filter(
        input_folder=combined_dir,
        output_folder=os.path.join(tmp_dir, "zipf-filtered"),
        csv_output_folder=os.path.join(tmp_dir, "zipf-csv"),
        glossary_output_folder=zipf_dir
    )
    compute_tf_idf(
        input_folder=combined_dir,
        output_scores_folder=os.path.join(tmp_dir, "tfidf-scores"),
        output_words_folder=tfidf_dir
    )

    # Step 4: Generate glossary JSON only for uploaded file
    glossary_out = os.path.join(tmp_dir, "glossary")
    os.makedirs(glossary_out, exist_ok=True)
    uploaded_base = os.path.splitext(file.filename)[0]
    generate_glossary(zipf_dir, tfidf_dir, "unused", glossary_out)

    # Load definitions for the uploaded file
    full_glossary = {}
    for fname in os.listdir(glossary_out):
        if not fname.endswith(".json"):
            continue
        base = os.path.splitext(fname)[0].replace("_glossary", "")
        if base != uploaded_base:
            continue
        path = os.path.join(glossary_out, fname)
        words = json.load(open(path, encoding="utf-8"))
        definitions = {word: define_word(word) for word in words}
        full_glossary[base] = definitions

    # --- ONLY THIS BLOCK CHANGED: write rich JSON into Firestore ---
    if full_glossary:
        coll_ref = db.collection("glossaries").document(run_id).collection(uploaded_base)
        for word, info in full_glossary[uploaded_base].items():
            coll_ref.add({
                "word":           word,
                "part_of_speech": info.get("part_of_speech", ""),
                "pronunciation":  info.get("pronunciation", ""),
                "definition":     info.get("definition", ""),
                "example":        info.get("example", ""),
                "translation":    info.get("translation", "")
            })
        return {
            "id":      run_id,
            "status":  "success",
            "words":   list(full_glossary[uploaded_base].keys()),
            "preview": [
                {"word": w, **full_glossary[uploaded_base][w]}
                for w in list(full_glossary[uploaded_base].keys())[:3]
            ]
        }
    else:
        return {"id": run_id, "status": "empty"}
