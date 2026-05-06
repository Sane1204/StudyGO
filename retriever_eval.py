from main import process_pdf
from query_engine import hybrid_retriever
from llama_index.retrievers.bm25 import BM25Retriever
import pandas as pd

def evaluate_retrieval(index, nodes, questions_df, top_k=5):
    results = []

    for _, row in questions_df.iterrows():
        query = row["user_input"]
        relevant_ids = set(str(row["relevant_chunk_ids"]).split(";"))

        vector_retriever = index.as_retriever()
        bm25 = BM25Retriever.from_defaults(nodes=nodes)

        retrieved_nodes = hybrid_retriever(vector_retriever, bm25, query, top_k=top_k)
        retrieved_ids = [node.node_id for node in retrieved_nodes]

        # Precision@K
        hits = [1 if nid in relevant_ids else 0 for nid in retrieved_ids]
        precision_at_k = sum(hits) / top_k

        # Recall@K
        recall_at_k = sum(hits) / len(relevant_ids) if relevant_ids else 0

        # MRR
        mrr = 0
        for rank, nid in enumerate(retrieved_ids, start=1):
            if nid in relevant_ids:
                mrr = 1 / rank
                break

        results.append({
            "query": query,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "mrr": mrr,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": list(relevant_ids)
        })

    return pd.DataFrame(results)

def evaluate_bm25_only(nodes,questions_df,top_k=5):
    results=[]

    for _, row in questions_df.iterrows():
        query = row["user_input"]
        relevant_ids = set(str(row["relevant_chunk_ids"]).split(";"))

        bm25 = BM25Retriever.from_defaults(nodes=nodes)
        retrieved = bm25.retrieve(query)

        retrieved_ids = [r.node.node_id for r in retrieved[:top_k]]
        print(f"BM25 retreived ids {retrieved_ids}")
        print(relevant_ids)

        hits = [1 if nid in relevant_ids else 0 for nid in retrieved_ids]
        precision_at_k = sum(hits) / top_k
        recall_at_k = sum(hits) / len(relevant_ids) if relevant_ids else 0
 
        mrr = 0
        for rank, nid in enumerate(retrieved_ids, start=1):
          if nid in relevant_ids:
            mrr = 1 / rank
            break
 
        results.append({
            "query": query,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "mrr": mrr,
        })
 
    return pd.DataFrame(results)

def evaluate_vector_only(index,questions_df,top_k=5):

    results=[]

    for _, row in questions_df.iterrows():
        query = row["user_input"]
        relevant_ids = set(str(row["relevant_chunk_ids"]).split(";"))
        retriever = index.as_retriever()
        retrieved = retriever.retrieve(query)

        retrieved_ids = [r.node.node_id for r in retrieved[:top_k]]

    
        hits = [1 if nid in relevant_ids else 0 for nid in retrieved_ids]
   
        precision_at_k = sum(hits) / top_k
        recall_at_k = sum(hits) / len(relevant_ids) if relevant_ids else 0
 
        mrr = 0
        for rank, nid in enumerate(retrieved_ids, start=1):
           if nid in relevant_ids:
                mrr = 1 / rank
                break
 
        results.append({
            "query": query,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "mrr": mrr,
        })
 
    return pd.DataFrame(results)

def print_results(label, df):
    print(f"\n=== {label} ===")
    print(f"Mean Precision@5: {df['precision@k'].mean()*100:.2f}%")
    print(f"Mean Recall@5:    {df['recall@k'].mean()*100:.2f}%")
    print(f"Mean MRR:         {df['mrr'].mean()*100:.2f}%")


def main():
    index, nodes = process_pdf(
        pdf="ai_intro.pdf",
        document_type="notes",
        file_name="ai_intro.pdf"
    )

    df = pd.read_csv("questions.csv")
    results = evaluate_retrieval(index, nodes, df, top_k=5)
    bm25_results=evaluate_bm25_only(nodes,df,top_k=5)
    vector_results=evaluate_vector_only(index,df,top_k=5)

    print_results("HYBRID", results)
    print_results("BM25 ONLY", bm25_results)
    print_results("VECTOR ONLY", vector_results)

    comparison = pd.DataFrame({
        "Retriever": ["Hybrid", "BM25 Only", "Vector Only"],
        "Precision@5": [
            results["precision@k"].mean()*100,
            bm25_results["precision@k"].mean()*100,
            vector_results["precision@k"].mean()*100,
        ],
        "Recall@5": [
            results["recall@k"].mean()*100,
            bm25_results["recall@k"].mean()*100,
            vector_results["recall@k"].mean()*100,
        ],
        "MRR": [
            results["mrr"].mean()*100,
            bm25_results["mrr"].mean()*100,
            vector_results["mrr"].mean()*100,
        ],
    })
    print("\n==== ABLATION COMPARISON TABLE ====")
    print(comparison.to_string(index=False))
 
    results.to_csv("hybrid_retrieval_results.csv", index=False)
    bm25_results.to_csv("bm25_retrieval_results.csv", index=False)
    vector_results.to_csv("vector_retrieval_results.csv", index=False)
    comparison.to_csv("ablation_comparison.csv", index=False)

    

    

if __name__ == "__main__":
    main()