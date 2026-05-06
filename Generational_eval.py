from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd 
import nltk
nltk.download('punkt')

from main import process_pdf
from ragas_evaluation import run_doubt_with_ref

def compute_score(reference,response):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge=scorer.score(reference,response)

    ref_token = [reference.split()]
    hyp_tokens= response.split()
    bleu= sentence_bleu(ref_token,hyp_tokens,smoothing_function=SmoothingFunction().method1)

    return {
        "rouge1": rouge['rouge1'].fmeasure,
        "rougeL":rouge['rougeL'].fmeasure,
        "bleu":bleu,    }

def main():
    index,nodes=process_pdf(
        pdf="ai_intro.pdf",
        document_type="Notes",
        file_name="ai_intro.pdf",
    )

    df = pd.read_csv("questions.csv")
    results=[]

    responses=[]
    references=[]

    for _,row in df.iterrows():
        result=run_doubt_with_ref(index,nodes,row["user_input"])
        response=result["response"]
        reference=row["reference"]

        score = compute_score(reference,response)
        score["user_input"]= row["user_input"]
        score["response"]= response
        score["reference"]= reference
        results.append(score)
        responses.append(response)
        references.append(reference)

    P,R,F1= bert_score(responses,references , lang="eng")
    for i , row in enumerate(results):
        row["bertscore_f1"]= F1[i].item()
    
    result_df=pd.DataFrame(results)
    print(result_df[["user_input","rouge1","rougeL","bleu","bertscore_f1"]])
    print("\n ===== AVERAGE =====")
    print(f"ROUGE-1: {result_df['rouge1'].mean():.3f}") 
    print(f"ROUGE-L: {result_df['rougeL'].mean():.3f}") 
    print(f"BLEU: {result_df['bleu'].mean():.3f}") 
    print(f"BERTScore: {result_df['bertscore_f1'].mean():.3f}") 

    result_df.to_csv("generation_eval_results.csv", index=False) 
    
if __name__ == "__main__": 
        main()