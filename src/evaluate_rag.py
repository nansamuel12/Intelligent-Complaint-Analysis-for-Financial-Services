import pandas as pd
import os
from rag_pipeline import ComplaintRAG

def evaluate():
    print("Starting RAG Evaluation...")
    
    # 1. Initialize RAG
    rag = ComplaintRAG()
    try:
        rag.load_resources()
    except Exception as e:
        print(f"Failed to load resources: {e}")
        return

    # 2. Define Test Questions
    questions = [
        "What are the most common complaints regarding Credit Cards?",
        "How do customers describe issues with money transfers?",
        "Can you find examples of incorrect fees charged on savings accounts?",
        "What are the complaints about unexpected interest rates?",
        "Describe a situation where a customer was harassed by debt collectors.",
        "What happened in complaint regarding 'incorrect information on your report'?", # Specific
        "How much time does it take to resolve a dispute?", # Might be hard to answer
        "Who is the CEO of the CFPB?", # Out of context -> Should say "I don't know"
        "Tell me about a complaint involving a stolen card.",
        "What solutions do companies offer for late fees?"
    ]
    
    results = []
    
    print(f"\nEvaluating {len(questions)} questions...")
    
    for q_idx, q in enumerate(questions):
        print(f"Processing Q{q_idx+1}: {q}")
        
        try:
            # Run RAG
            output = rag.answer_question(q, k=5)
            
            # Extract main source (top 1) for brevity in table
            top_source = output['sources'][0] if output['sources'] else {}
            source_summary = f"ID: {top_source.get('complaint_id', 'N/A')}, Prod: {top_source.get('product', 'N/A')}"
            
            results.append({
                "Question": q,
                "Generated Answer": output['answer'],
                "Top Source ID": top_source.get('complaint_id'),
                "Context Retrieved (Chars)": len(output.get('context_used', "")),
                # Placeholders for manual evaluation
                "Accuracy (1-5)": "",
                "Relevance (1-5)": "",
                "Hallucinations": "",
                "Notes": ""
            })
            
        except Exception as e:
            print(f"Error on Q{q_idx+1}: {e}")
            results.append({
                "Question": q,
                "Generated Answer": "ERROR",
                "Notes": str(e)
            })

    # 3. Create DataFrame
    df_eval = pd.DataFrame(results)
    
    # 4. Save
    output_path = "data/rag_evaluation.csv"
    df_eval.to_csv(output_path, index=False)
    print(f"\nEvaluation complete. Results saved to {output_path}")
    print("\nPlease open the CSV and fill in the manual evaluation scores.")

if __name__ == "__main__":
    evaluate()
