import os
import json
import logging
import faiss
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_dense_retrieval():
    """
    Implements the dense retrieval pipeline using Sentence-Transformers and FAISS.
    1. Loads data using beir.GenericDataLoader.
    2. Loads S-BERT model.
    3. Encodes the corpus and builds a FAISS index.
    4. Encodes queries and searches the index for top-100.
    5. Saves results in the required JSON format.
    """
    data_path = "datasets/scifact"
    model_name = "all-MiniLM-L6-v2"
    k_top = 100

    # --- 1. Load Data ---
    logger.info(f"Loading test split from {data_path} using BEIR...")
    try:
        corpus, queries, _ = GenericDataLoader(data_path).load(split="test")
    except Exception as e:
        logger.error(f"Failed to load data. Did you run 'python download_data.py' first? Error: {e}")
        return

    # --- 2. Load Model ---
    logger.info(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)

    # --- 3. Index Corpus with FAISS ---
    logger.info("Preparing corpus for encoding...")
    doc_ids = list(corpus.keys())
    corpus_texts = [
        corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
        for doc_id in doc_ids
    ]
    
    logger.info(f"Encoding {len(corpus_texts)} documents... (This will take time)")
    corpus_embeddings = model.encode(corpus_texts, 
                                     show_progress_bar=True, 
                                     convert_to_numpy=True)
    
    corpus_embeddings = corpus_embeddings.astype('float32')
    
    logger.info("Building FAISS index...")
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings)
    logger.info(f"FAISS index built. Total documents indexed: {index.ntotal}")

    # --- 4. Retrieve Top-100 ---
    logger.info("Encoding queries and running search...")
    results = {}
    
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_embeddings = model.encode(query_texts, 
                                    show_progress_bar=True, 
                                    convert_to_numpy=True)
    
    query_embeddings = query_embeddings.astype('float32')
    faiss.normalize_L2(query_embeddings)
    
    D, I = index.search(query_embeddings, k_top)
    
    logger.info("Formatting results...")
    for i, query_id in enumerate(query_ids):
        query_results = {}
        for j in range(k_top):
            doc_index = I[i][j]
            doc_id = doc_ids[doc_index]
            score = D[i][j]
            query_results[doc_id] = float(score)
            
        results[query_id] = query_results

    # --- 5. Save Results ---
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dense_results.json")
    
    logger.info(f"Saving retrieval results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info("Dense retrieval finished successfully.")

if __name__ == "__main__":
    run_dense_retrieval()