# RAG Pipeline over a PDF

A retrieval-augmented generation (RAG) pipeline built during my Generative AI
internship at EY (Sep - Nov 2024). It answers questions about a document by
retrieving the most relevant chunks and passing them to an LLM, so answers are
grounded in the document rather than the model's general knowledge.

The internship focus was **Document Intelligence for Gen AI**, and the central
question was a practical one for a firm handling sensitive client data:
**can you run RAG entirely locally (data never leaves the machine), and what does
that cost you in speed?** So the project was built two ways — fully local, and
cloud-accelerated - and the two were benchmarked against each other.

Everything runnable lives in one notebook: **[rag_pipeline.ipynb](rag_pipeline.ipynb)**.
The full internship write-up - methodology, embedding-model exploration, and the
latency benchmark - is in **[internship-report.pdf](internship-report.pdf)**.


## Stack

- **PyMuPDF** - extract text from the PDF
- **LangChain** - chunking + the RetrievalQA chain
- **sentence-transformers (Stella)** - **local** embeddings, so document text is
  never sent to a third-party API
- **Chroma** - local, on-disk vector database
- **LLM** - local **Llama 3.1 (8B) via Ollama** for full privacy, or
  **Groq** - hosted Llama 3.1 for speed

## Why these choices

- **Chunking (`CharacterTextSplitter`, size / overlap).** Embedding smaller
  chunks instead of whole documents means a query retrieves only the most
  relevant segments — fewer input tokens and more targeted context for the LLM.
  Overlap keeps ideas that straddle a chunk boundary retrievable from either side.
- **Stella embeddings (`bijaygurung/stella_en_400M_v5`).** Chosen from the
  [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) because it
  embeds locally — keeping sensitive data on-device. I downsized from
  `stella_en_1.5B_v5` (8192-dim) to the 400M/1024-dim variant to cut resource use
  while keeping quality, and Stella was picked partly for its flexible embedding
  dimensions (256 → 8192), a lever for trading accuracy against speed.
- **Chroma.** An open-source, on-disk vector store - no external service or
  account, which is what makes this clone-and-run.

## Local vs. cloud: the benchmark

Same pipeline, same query ("summarize the document"), three ways to run the
generation step. Measured during the internship on a 16 GB Windows machine:

| Setup                                   | Hardware      | Time      | Data privacy                    |
|-----------------------------------------|---------------|-----------|---------------------------------|
| Local Stella + local Llama 3.1          | GPU           | **8.69 min** | Full — nothing leaves the machine |
| Local pipeline (CPU, no separate load)  | CPU only      | **3.62 min** | Full                            |
| Groq-hosted Llama 3.1                    | API           | **2.15 s**  | Data sent to cloud API          |

The takeaway that shaped the project: **local inference gives you complete data
isolation but is far too slow for interactive use; Groq is ~200× faster but
trades away that isolation.** For a firm handling client data that tradeoff is
the whole decision.

## What I learned / would improve

- **The local-vs-cloud tradeoff is a real product decision**, not a detail — data
  security vs. latency, quantified above.
- **Retrieval quality matters more than the LLM choice** — chunk size and `k`
  moved answer quality more than swapping the model did.
- **Hardware reality bites**: running Llama 3.1 (8B) locally meant CUDA/xformers
  version conflicts, 16 GB RAM kernel crashes, and a ~60 GB model clone — which is
  exactly why the local path was migrated to Windows and why Groq was worth it.
- **Would improve**: a smarter splitter that respects headings, moving off the
  legacy `RetrievalQA` to an LCEL chain, and a small set of test questions to
  measure retrieval quality instead of judging by eye.
