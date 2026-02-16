# YouTube CS Knowledge Base: High-Density RAG

An advanced Retrieval-Augmented Generation (RAG) system designed to provide precise, timestamped educational support across a massive library of 10,000+ computer science lectures from top universities and platforms like FreeCodeCamp.

## 🚀 Project Overview

Most RAG systems suffer from "context dilution" and "retrieval noise" when processing raw, long-form technical transcripts. This project implements a **High-Density Semantic Indexing** architecture, utilizing statistical de-noising to ensure that only high-value instructional content is indexed, significantly improving retrieval precision and LLM generation quality.

### Key Achievements (Phase 1 & 2 Complete)

* **Massive Ingestion:** Built a resilient pipeline to ingest and process **10,000+ video IDs** for technical courses longer than 5 minutes.
* **Resilient Data Ingestion:** Developed a robust request-handling layer to navigate platform rate-limits and infrastructure constraints, ensuring stable high-volume transcript extraction.
* **Statistical Quality Pipeline (Perplexity Filtering):** Engineered a granular cleaning engine using **GPT-2 Small** to calculate sentence-level perplexity. This automatically prunes:
* **Conversational Noise:** Administrative chatter ("Can you hear me?", "Check the chat").
* **Boilerplate:** Repetitive intros/outros and "Like/Subscribe" calls to action.
* **ASR Errors:** Garbled speech-to-text segments and non-semantic "word salad."


* **NLP Restoration:** Utilized **DeepMultilingualPunctuation** and custom Regex suites to restore semantic structure and technical syntax to raw, unpunctuated subtitles.

---

## 🛠️ Architecture & Tech Stack

* **Framework:** LangChain (Orchestration)
* **Vector Database:** Qdrant (Storage & Similarity Search)
* **LLMs:** Llama-3-8B (Planned for Generation)
* **Statistical Filtering:** GPT-2 Small (Perplexity Scoring)
* **Preprocessing:** Python, spaCy, BERT-based Punctuation Models
* **Indexing Strategy:** High-Density Semantic Chunks (Pruned sentence-level blocks)

---

## 📂 Data Pipeline

The project follows a **JSON Staging Strategy** to move from raw audio transcripts to a refined "Golden Dataset":

1. **Ingestion:** Extraction of raw YouTube JSON (ID, Timestamp, Text).
2. **Normalization:** Text cleaning, punctuation restoration, and timestamp alignment via Sequence Matching.
3. **Statistical Pruning:** Sentence-level perplexity analysis to strip "low-information" segments.
4. **Indexing:** Pushing high-density instructional units into Qdrant with detailed metadata for video source and timing.

---

## 🛠️ How to Use

> **Note:** The core data engineering and statistical cleaning engines are currently finalized. Once the retrieval and generation modules are fully integrated, a detailed "How to Run" section with Docker and local environment setup will be added here.

Training Logs and code are still in Kaggle now. Will update soon. 
