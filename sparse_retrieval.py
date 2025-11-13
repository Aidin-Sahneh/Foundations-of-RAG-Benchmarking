import os
import json
import logging
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_bm25_retrieval():
    """
    Implements the sparse retrieval pipeline using BM25.
    1. Loads data using beir.GenericDataLoader.
    2. Tokenizes and indexes the corpus.
    3. Retrieves top-100 documents for each query.
    4. Saves results in the required JSON format.
    """
    data_path = "datasets/scifact"
    
    # --- 1. Load Data ---
    logger.info(f"Loading test split from {data_path} using BEIR...")
    try:
        corpus, queries, _ = GenericDataLoader(data_path).load(split="test")
    except Exception as e:
        logger.error(f"Failed to load data. Did you run 'python download_data.py' first? Error: {e}")
        return

    # --- 2. Index with BM25 ---
    logger.info("Preparing corpus for BM25...")
    doc_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).lower()
        for doc_id in doc_ids
    ]
    tokenized_corpus = [doc.split() for doc in corpus_texts]
    
    logger.info("Indexing corpus with BM25... (This may take a moment)")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # --- 3. Retrieve Top-100 ---
    logger.info("Running retrieval for all queries...")
    results = {}
    k_top = 100
    
    for query_id, query_text in tqdm(queries.items(), desc="Processing queries"):
        tokenized_query = query_text.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[::-1][:k_top]
        
        query_results = {doc_ids[idx]: float(doc_scores[idx]) for idx in top_k_indices}
        results[query_id] = query_results

    # --- 4. Save Results ---
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sparse_results.json")
    
    logger.info(f"Saving retrieval results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info("Sparse retrieval finished successfully.")

if __name__ == "__main__":
    run_bm25_retrieval()