# 🤗 Gemma Recipes — Fine-Tuning, Inference, and RAG Examples

Welcome to the **Gemma Recipes** repository! This project contains useful examples for working with the [Gemma family of models](https://ai.google.dev/gemma) including inference, fine-tuning, and retrieval-augmented generation (RAG).

---

## 📘 Notebooks Included

### 🔹 `Gemma_RAG.ipynb`

A complete example demonstrating **Retrieval-Augmented Generation (RAG)** using Gemma models. This notebook walks through:

- Encoding custom text snippets using `SentenceTransformer`
- Creating a semantic index with **FAISS**
- Performing semantic search on local data using vector similarity
- Retrieving relevant context and running queries

#### 🔍 Core Features

| Component | Description |
|----------|-------------|
| **Embedding Model** | `all-MiniLM-L6-v2` from SentenceTransformers |
| **Semantic Search** | Euclidean-based FAISS indexing |
| **Retrieval** | Top-k text snippet matching using query vectors |
| **Gemma Integration** | Intended for use with Gemma models for RAG pipelines |

---

## 📦 Getting Started

Install required dependencies:

```bash
pip install -Uq sentence-transformers transformers accelerate faiss-cpu timm numpy



