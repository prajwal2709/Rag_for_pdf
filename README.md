# 📚 RAG for PDF

> An agentic Retrieval-Augmented Generation (RAG) application that lets you upload documents, build a semantic knowledge base, and ask questions grounded in the uploaded content.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/DeepAgents-Agentic%20AI-111827?style=for-the-badge" alt="DeepAgents">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenRouter-LLM%20%2B%20Embeddings-6B46C1?style=for-the-badge" alt="OpenRouter">
</p>

---

## ✨ Overview

**RAG for PDF** is a document question-answering application built around the Retrieval-Augmented Generation architecture.

Instead of asking an LLM to answer entirely from its pretrained knowledge, the application first processes the user's document, creates semantic embeddings, retrieves the most relevant document chunks, and uses those retrieved passages as context for generating an answer.

The project also explores an **agentic RAG workflow using DeepAgents**, where the agent can search the indexed document and delegate retrieved chunks to a specialized analysis subagent.

### Core idea

```text
Upload Document
      ↓
Extract Text
      ↓
Split into Chunks
      ↓
Generate Embeddings
      ↓
Semantic Vector Search
      ↓
Retrieve Relevant Chunks
      ↓
Deep Agent
      ↓
Chunk Analysis
      ↓
Grounded Answer
