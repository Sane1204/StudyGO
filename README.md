# StudyGO — Local RAG Study Assistant

## Requirements
- Python 3.10+
- Ollama installed and running locally
- Models pulled: llama3, nomic-embed-text,qwen2.5

## Installation
pip install llama-index streamlit ragas bert-score 
rouge-score nltk pandas langchain-openai

## Running the Application
streamlit run app.py

## Running Evaluations
- Retrieval: python retriever_eval.py
- Generational: python Generational_eval.py
- No-RAG Baseline: python norag_baselineeval.py
- RAGAS: python ragas_evaluation.py
- Comparison: python rag_no_rag_comp.py

## Supported File Formats
PDF, DOCX, PPTX, TXT, MD