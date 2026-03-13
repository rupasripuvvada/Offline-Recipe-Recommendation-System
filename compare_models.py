import pandas as pd
import sqlite3
import json
import os

from recommender import (
    evaluate_recommendation_system,
    compute_scores,
    df as proposed_df,
    load_inventory as prop_load_inventory
)

from baseline_model import run_baseline_evaluation


def load_baseline_inventory():
    """Load inventory from database for baseline model."""
    
    conn = sqlite3.connect("grocery_ocr.db")
    df_inv = pd.read_sql_query("SELECT item, expiry_date FROM inventory", conn)
    conn.close()

    if df_inv.empty:
        return [], {}

    df_inv["item"] = df_inv["item"].str.lower().str.strip()
    df_inv["expiry_date"] = pd.to_datetime(df_inv["expiry_date"], errors="coerce")

    today = pd.Timestamp.now().normalize()
    df_inv["days_left"] = (df_inv["expiry_date"] - today).dt.days

    df_inv = df_inv.groupby("item", as_index=False)["days_left"].min()

    baseline_dict = dict(zip(df_inv["item"], df_inv["days_left"]))
    baseline_items = list(baseline_dict.keys())

    return baseline_items, baseline_dict


def main():

    print("========================================")
    print("Shelf AI Recommendation Model Evaluation")
    print("========================================")

    print("\nLoading proposed inventory format...")
    prop_items, prop_dict, prop_text = prop_load_inventory()

    print("Loading baseline inventory from database...")
    baseline_items, baseline_dict = load_baseline_inventory()

    print("Loading recipe dataset...")
    df_raw = pd.read_csv("Final_Cleaned_Dataset.csv")

    print("\nRunning Baseline Model Evaluation...")
    baseline_metrics = run_baseline_evaluation(
        df_raw.copy(),
        baseline_items,
        baseline_dict
    )

    print("\nRunning Proposed Model Evaluation...")
    scores_df = compute_scores(
        proposed_df.copy(),
        prop_items,
        prop_dict,
        prop_text
    )

    proposed_metrics = evaluate_recommendation_system(scores_df, k=50)

    # Round metrics for clean display
    for m in [baseline_metrics, proposed_metrics]:
        for key in m:
            if isinstance(m[key], float):
                m[key] = round(m[key], 3)

    results = {
        "baseline": baseline_metrics,
        "proposed": proposed_metrics
    }

    os.makedirs("static", exist_ok=True)
    output_path = "static/comparison_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation Complete!")
    print(f"Metrics saved to: {output_path}")

    print("\n===== RESULTS SUMMARY =====")

    print("\nBaseline Metrics:")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v}")

    print("\nProposed Metrics:")
    for k, v in proposed_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()