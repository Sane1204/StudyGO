import pandas as pd

rag = pd.read_csv("generation_eval_results.csv")
no_rag = pd.read_csv("no_rag_baseline_results.csv")

comparison = pd.DataFrame({
    "System": ["No-RAG Baseline", "RAG System"],
    "ROUGE-1": [no_rag["rouge1"].mean(), rag["rouge1"].mean()],
    "ROUGE-L": [no_rag["rougeL"].mean(), rag["rougeL"].mean()],
    "BLEU":    [no_rag["bleu"].mean(), rag["bleu"].mean()],
    "BERTScore": [no_rag["bertscore_f1"].mean(), rag["bertscore_f1"].mean()],
})

print(comparison.to_string(index=False))
comparison.to_csv("rag_vs_no_rag_comparison.csv", index=False)