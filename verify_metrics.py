import pandas as pd
import numpy as np
from recommender import evaluate_recommendation_system

def test_evaluation():
    # Create a mock scores_df with necessary columns
    data = {
        "Final_Score": np.random.rand(100),
        "Coverage": np.random.rand(100),
        "Expiry": np.random.rand(100),
        "Matched": [["apple"]] * 100
    }
    df = pd.DataFrame(data)
    
    k = 50
    metrics = evaluate_recommendation_system(df, k=k)
    
    print("\n" + "="*60)
    print("📊 RESEARCH-LEVEL SYSTEM EVALUATION REPORT")
    print("="*60)

    print("\n🚀 SUMMARY METRICS")
    print("-" * 20)
    print(f"NDCG@8:      {metrics['ndcg_8']:<10} (Measures ranking quality of the top 8 results)")
    print(f"F1 (Overall): {metrics['f1_overall']:<10} (Balance between Precision and Recall for relevance)")

    print("\n▲ RANKING METRICS (Evaluating the top {k} recommendations)")
    print("-" * 20)
    print(f"Precision@{k}: {metrics['precision_k']:<10} (Ratio of relevant recipes in the top {k})")
    print(f"Recall@{k}:    {metrics['recall_k']:<10} (Ratio of all relevant recipes captured in the top {k})")
    print(f"F1@{k}:        {metrics['f1_k']:<10} (Harmonic mean of Precision and Recall at K)")
    print(f"MRR:          {metrics['mrr']:<10} (Mean Reciprocal Rank - how soon the first relevant result appears)")
    print(f"NDCG@{k}:      {metrics['ndcg_k']:<10} (Normalized Discounted Cumulative Gain - accounts for position relevance)")

    print("\n🤖 CLASSIFICATION PERFORMANCE")
    print("-" * 20)
    print(f"Precision:    {metrics['precision']:<10} (Accuracy of positive predictions)")
    print(f"Recall:       {metrics['recall']:<10} (Ability to find all positive instances)")
    print(f"F1 Score:     {metrics['f1']:<10} (Overall classifier accuracy)")
    
    print("\n📦 CONFUSION MATRIX (Relevance Diagnostics)")
    print(f"   TN: {metrics['tn']:<5} FP: {metrics['fp']:<5}")
    print(f"   FN: {metrics['fn']:<5} TP: {metrics['tp']:<5}")
    print("   (TN=True Neg, FP=False Pos, FN=False Neg, TP=True Pos)")

    print("\n🌱 SUSTAINABILITY & LOGISTICS")
    print("-" * 20)
    print(f"Catalog Coverage:           {metrics['catalog_coverage']:<10} (Market reach - % of recipes matchable)")
    print(f"Food Waste Reduction Index: {metrics['fwri']:<10} (Efficiency in using items nearing expiry)")

    print("\n" + "="*60)
    print("VERIFICATION SUCCESSFUL - ALL METRICS ACCOUNTED FOR")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_evaluation()
