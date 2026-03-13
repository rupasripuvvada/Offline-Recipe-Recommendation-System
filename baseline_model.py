import pandas as pd
import numpy as np
import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender import evaluate_recommendation_system

IGNORE_INGREDIENTS = set(["water", "sunflower oil", "salt", "oil"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_recipes(df):
    df = df.copy()
    if "Cleaned_Ingredients" in df.columns and isinstance(df["Cleaned_Ingredients"].iloc[0], str):
        df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(eval)
    
    df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(
        lambda x: [clean_text(i) for i in x] if isinstance(x, list) else []
    )
    df["ingredient_text"] = df["Cleaned_Ingredients"].apply(lambda x: " ".join(x))
    return df

def run_baseline_evaluation(df_raw, inventory_items, inventory_dict):
    df_clean = preprocess_recipes(df_raw)
    
    inventory_items_filtered = [item for item in inventory_items if item not in IGNORE_INGREDIENTS]
    inventory_text_filtered = " ".join(inventory_items_filtered)
    
    vectorizer = TfidfVectorizer()
    corpus = df_clean["ingredient_text"].tolist() + [inventory_text_filtered]
    vectorizer.fit(corpus)
    inventory_vec = vectorizer.transform([inventory_text_filtered])
    
    matched_list, missed_list, coverage_list, expiry_list, semantic_list = [], [], [], [], []

    for idx, row in df_clean.iterrows():
        recipe_vec = vectorizer.transform([row["ingredient_text"]])
        sim = cosine_similarity(recipe_vec, inventory_vec)[0][0]

        ingredients = [i for i in row["Cleaned_Ingredients"] if i not in IGNORE_INGREDIENTS]
        matched = [i for i in ingredients if i in inventory_items_filtered]
        missed = [i for i in ingredients if i not in inventory_items_filtered]
        coverage = len(matched) / max(len(ingredients), 1)

        expiry_scores = []
        for i in matched:
            days = inventory_dict.get(i, 30)
            normalized = max(0, (30 - days) / 30)
            expiry_scores.append(normalized)
        expiry = np.mean(expiry_scores) if expiry_scores else 0

        matched_list.append(matched)
        missed_list.append(missed)
        coverage_list.append(round(coverage,3))
        expiry_list.append(round(expiry,3))
        semantic_list.append(round(sim,3))

    df_clean["Matched"] = matched_list  # Renamed for evaluator compatibility
    df_clean["Missed_Ingredients"] = missed_list
    df_clean["Coverage"] = coverage_list # Renamed for evaluator compatibility
    df_clean["Expiry"] = expiry_list # Renamed for evaluator compatibility
    df_clean["Semantic_Score"] = semantic_list

    # Default logic is 0.4 Sem + 0.4 Cov + 0.2 Exp
    df_clean["Final_Score"] = (
        0.3 * df_clean["Semantic_Score"] +
        0.5 * df_clean["Coverage"] +
        0.2 * df_clean["Expiry"]
    )
    
    # Evaluate
    # use k=50 just like in verify_metrics.py
    metrics = evaluate_recommendation_system(df_clean, k=50)
    return metrics
