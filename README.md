
# Implementation Challenge

This project implements and evaluates two fundamental information retrieval systems, a sparse retriever (BM25) and a dense retriever (Sentence-BERT + FAISS). The systems are benchmarked on the `SciFact` dataset from the BEIR library.

## Project Structure


The repository is structured as follows:


```
├── datasets/ # Directory for downloaded BEIR datasets (e.g., scifact)

├── results/ # Output directory for retrieval results

│ ├── sparse_results.json

│ └── dense_results.json

├── report/ 
│   └── Literature_Review_Part1.pdf

├── venv/ # Python virtual environment (ignored by git)

├── .gitignore

├── download_data.py # Script to download the SciFact dataset

├── sparse_retrieval.py # Script to run and evaluate BM25 retrieval

├── dense_retrieval.py # Script to run and evaluate S-BERT + FAISS retrieval

├── evaluation.py # The evaluation script

├── requirements.txt # Project dependencies

└── README.md 
```


## Setup and Installation

Follow these steps to set up the environment and run the project.

**1. Clone the Repository (or download the files):**
```
git clone https://github.com/Aidin-Sahneh/Foundations-of-RAG-Benchmarking.git
cd Foundations-of-RAG-Benchmarking
```

**2\. Create a Python Virtual Environment:**


```
python -m venv venv

```

**3\. Activate the Environment:**

-   **Windows (PowerShell/CMD):**



    ```
    .\venv\Scripts\Activate

    ```

-   **macOS/Linux:**

    

    ```
    source venv/bin/activate

    ```

**4\. Install Dependencies:**


```
pip install -r requirements.txt

```

* * * * *

How to Run
----------

The pipeline must be run in the following order.

### Step 1: Download the Data

This script downloads the `scifact` dataset from the BEIR benchmark into the `datasets/` directory.



```
python download_data.py

```

### Step 2: Run Sparse Retrieval (BM25)

This script loads the data, indexes the corpus using `rank-bm25`, retrieves the top 100 documents for each query, and saves the output to `results/sparse_results.json`.



```
python sparse_retrieval.py

```

### Step 3: Run Dense Retrieval (S-BERT + FAISS)

This script loads the data, embeds the entire corpus using `all-MiniLM-L6-v2`, builds a FAISS index, retrieves the top 100 documents for each query, and saves the output to `results/dense_results.json`.



```
python dense_retrieval.py

```

### Step 4: Run Evaluation

Use the provided `evaluation.py` script to evaluate the generated results files against the official qrels.

**To evaluate Sparse Retrieval (BM25):**



```
python evaluation.py datasets/scifact results/sparse_results.json

```

**To evaluate Dense Retrieval (S-BERT):**



```
python evaluation.py datasets/scifact results/dense_results.json

```

* * * * *

Results and Discussion
----------------------

The evaluation scripts produce the following scores for the `scifact` test split.

### Results Summary

| **Metric** | **🔺 Dense Retriever (S-BERT)** | **🔻 Sparse Retriever (BM25)** |
| --- | --- | --- |
| **NDCG@10** | **0.64508** | 0.55970 |
| **Recall@100** | **0.92500** | 0.79294 |
| **MAP@100** | **0.60307** | 0.52019 |

### Discussion

#### 1\. Which retriever performed better?

As the results table clearly shows, the **Dense Retriever (S-BERT + FAISS) significantly outperformed the Sparse Retriever (BM25)** across all major evaluation metrics. The most notable improvements are a **15.2% increase in NDCG@10** (quality of top 10 results) and a **16.6% increase in Recall@100** (ability to find all relevant documents).

#### 2\. Why do you think that is?

The superior performance of the dense retriever is primarily due to its ability to overcome the **"lexical gap"** through **semantic understanding**.

-   **Sparse Retrieval (BM25)** is based on *keyword matching*. It fails if a query and a relevant document use different words (synonyms, paraphrasing) to describe the same concept.

-   **Dense Retrieval (S-BERT)** operates on *semantic meaning*. The `SciFact` dataset requires matching scientific claims (queries) to abstracts containing supporting evidence (corpus). This task is highly semantic. For example, a claim might state "X reduces Y," while the evidence says "Y is decreased by X." BM25 would struggle with this, whereas S-BERT, having been trained on vast text corpora, understands that these two phrases are semantically equivalent.

The dense model captures the *intent* of the query rather than just the literal words, which is essential for a fact-checking task.

#### 3\. What are the performance trade-offs?

While the dense retriever provided superior results, it comes with different performance trade-offs compared to BM25.

-   **Retrieval Quality:** **Dense wins.** The results speak for themselves.

-   **Indexing Cost:** **BM25 wins.**

    -   **BM25:** Indexing is extremely fast, CPU-based, and requires minimal memory. It tokenizes the corpus, which is a very cheap operation.

    -   **Dense:** Indexing is computationally expensive. It requires a powerful GPU (or significant CPU time) to run a forward pass of the SentenceTransformer model for *every single document* in the corpus (5,183 documents in this case).

-   **Search Speed:**

    -   **BM25:** Very fast for most use cases.

    -   **Dense (with FAISS):** After the index is built, searching is *extremely* fast. FAISS is highly optimized for vector similarity search, making retrieval (a single vector-matrix multiplication) almost instantaneous.

-   **Memory & Resources:** **BM25 wins.**

    -   **BM25:** The index is lightweight.

    -   **Dense:** Requires the S-BERT model (e.g., ~90MB for `all-MiniLM-L6-v2`) and the entire FAISS index (all document embeddings) to be loaded into memory, making its resource footprint significantly larger.

In summary, the dense retriever offers a significant boost in retrieval quality by understanding semantics, but this comes at the cost of a much more expensive and resource-intensive indexing process.

---
## Final Package

The final package contains both parts of the quiz:

* **Part 1: Literature Review** is available as a PDF document inside the **`report/`** directory.
* **Part 2: Implementation Challenge** consists of the Python scripts, `requirements.txt`, and this `README.md` file.
