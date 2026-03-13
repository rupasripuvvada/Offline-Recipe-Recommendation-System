import pandas as pd
import numpy as np
import sqlite3
import re
import spacy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -----------------------------
# Load models
# -----------------------------

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

IGNORE_INGREDIENTS = {"water", "salt", "oil", "sunflower oil"}


# -----------------------------
# Ingredient normalization
# -----------------------------

def normalize_ingredient(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    doc = nlp(text)
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop
    ]
    return " ".join(lemmas)

# -----------------------------
# Data Normalization
# -----------------------------

def normalize_cuisine(text):
    text = str(text).lower().strip()
    cuisine_map = {
        "andhra": "Andhra",
        "hyderabadi": "Hyderabadi",
        "south indian": "South Indian",
        "north indian": "North Indian",
        "bengali": "Bengali",
        "gujarati": "Gujarati",
        "indian": "Indian",
        "karnataka": "Karnataka",
        "kashmiri": "Kashmiri",
        "kerala": "Kerala",
        "lucknowi": "Lucknowi",
        "maharashtrian": "Maharashtrian",
        "nagaland": "Nagaland",
        "uttar pradesh": "Uttar Pradesh",
        "udupi": "Udupi",
        "tamil nadu": "Tamil Nadu",
        "rajasthani": "Rajasthani",
        "punjabi": "Punjabi",
        "himachal": "Himachal",
        "jharkhand": "Jharkhand",
        "oriya": "Odia"
    }

    for key in cuisine_map:
        if key in text:
            return cuisine_map[key]
    return "Other"

def normalize_meal(text):
    text = str(text).lower().strip()
    if "breakfast" in text:
        return "Breakfast"
    elif "lunch" in text:
        return "Lunch"
    elif "dinner" in text:
        return "Dinner"
    elif any(word in text for word in ["main course", "curry", "biryani", "rice", "roti", "paratha"]):
        return "Main Course"
    elif any(word in text for word in ["side dish", "accompaniment", "salad", "raita"]):
        return "Side Dish"
    elif any(word in text for word in ["dessert", "sweet", "cake", "halwa", "kheer"]):
        return "Dessert"
    elif any(word in text for word in ["snack", "starter", "appetizer", "chaat", "pakora"]):
        return "Snack"
    else:
        return "Other"

# -----------------------------
# Load inventory
# -----------------------------

def load_inventory():
    conn = sqlite3.connect("grocery_ocr.db")
    # Using grocery_items table now
    df_inv = pd.read_sql_query("SELECT item, expiry_date FROM grocery_items", conn)
    conn.close()

    if df_inv.empty:
        return [], {}, ""

    df_inv["expiry_date"] = pd.to_datetime(df_inv["expiry_date"], errors='coerce')
    df_inv = df_inv.dropna(subset=['expiry_date'])
    
    today = pd.Timestamp.now().normalize()
    df_inv["days_left"] = (df_inv["expiry_date"] - today).dt.days
    df_inv["item_norm"] = df_inv["item"].apply(normalize_ingredient)
    df_inv = df_inv.groupby("item_norm", as_index=False)["days_left"].min()

    inventory_dict = dict(zip(df_inv["item_norm"], df_inv["days_left"]))
    inventory_items = list(inventory_dict.keys())
    inventory_text = " ".join(inventory_items)

    return inventory_items, inventory_dict, inventory_text

# -----------------------------
# Fuzzy matching
# -----------------------------

def find_inventory_match(ingredient, inventory_keys):
    match = process.extractOne(
        ingredient,
        inventory_keys,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=70
    )
    return match[0] if match else None

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("Final_Cleaned_Dataset.csv")

if "Cuisine" in df.columns:
    df["Cuisine_Clean"] = df["Cuisine"].apply(normalize_cuisine)
elif "Cuisine_Clean" not in df.columns:
    df["Cuisine_Clean"] = "Other"

if "Course" in df.columns:
    df["MealType"] = df["Course"].apply(normalize_meal)
elif "MealType" not in df.columns:
    df["MealType"] = "Other"

df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(eval)
df["recipe_text"] = df["Cleaned_Ingredients"].apply(lambda x: " ".join(x))

recipe_embeddings = bi_encoder.encode(
    df["recipe_text"].tolist(),
    convert_to_numpy=True
)

# -----------------------------
# Scoring logic
# -----------------------------

def compute_scores(df_in, inventory_items, inventory_dict, inventory_text):
    """
    Computes scores for a given DataFrame of recipes and inventory.
    """
    if not inventory_items:
        df_in["Final_Score"] = 0
        return df_in

    inventory_emb = bi_encoder.encode(inventory_text, convert_to_numpy=True)
    all_sims = cosine_similarity(recipe_embeddings, [inventory_emb]).flatten()

    df_out = df_in.copy()
    df_out["Semantic_Score"] = all_sims[df_out.index]

    matched_list = []
    missed_list = []
    coverage_list = []
    expiry_list = []

    for idx, row in df_out.iterrows():
        ingredients = row["Cleaned_Ingredients"]
        matched = []
        missed = []
        expiry_scores = []

        for ing in ingredients:
            if ing.lower() in IGNORE_INGREDIENTS:
                continue
            match = find_inventory_match(ing, inventory_dict.keys())
            if match:
                matched.append(match)
                days = inventory_dict.get(match, 30)
                expiry_scores.append(max(0, (30 - days) / 30))
            else:
                missed.append(ing)

        matched = list(set(matched))
        missed = list(set(missed))
        coverage = len(matched) / len(ingredients) if ingredients else 0
        expiry = np.mean(expiry_scores) if expiry_scores else 0

        matched_list.append(matched)
        missed_list.append(missed)
        coverage_list.append(round(coverage, 3))
        expiry_list.append(round(expiry, 3))

    df_out["Matched"] = matched_list
    df_out["Missed"] = missed_list
    df_out["Coverage"] = coverage_list
    df_out["Expiry"] = expiry_list

    df_out["Final_Score"] = (
        0.30 * df_out["Semantic_Score"]
        + 0.50 * df_out["Coverage"]
        + 0.20 * df_out["Expiry"]
    )
    
    return df_out

# -----------------------------
# Recommend recipes
# -----------------------------

def recommend_recipes(cuisine="Any", diet="Any", meal="Any", offset=0):
    inventory_items, inventory_dict, inventory_text = load_inventory()
    if not inventory_items:
        return []

    # Get cooked history for last 5 days
    from datetime import datetime, timedelta
    five_days_ago = (datetime.now() - timedelta(days=5)).date()
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT recipe_name FROM cooked_history WHERE cooked_date >= ?", (five_days_ago,))
    cooked_recently = {row[0] for row in cur.fetchall()}
    conn.close()

    def get_filtered_df(df_in, c, d, m):
        lvl1 = df_in.copy()
        if c != "Any":
            lvl1 = lvl1[lvl1["Cuisine_Clean"].str.lower() == c.lower()]
        if d != "Any":
            lvl1 = lvl1[lvl1["Diet"].str.lower() == d.lower()]
        if m != "Any":
            lvl1 = lvl1[lvl1["MealType"].str.lower() == m.lower()]
        
        if not lvl1.empty:
            return lvl1, "Matched Cuisine + Diet + Meal"

        if d != "Any" and m != "Any":
            lvl2 = df_in[(df_in["MealType"].str.lower() == m.lower()) & (df_in["Diet"].str.lower() == d.lower())]
            if not lvl2.empty:
                return lvl2, "Matched Meal + Diet"

        if m != "Any":
            lvl3 = df_in[df_in["MealType"].str.lower() == m.lower()]
            if not lvl3.empty:
                return lvl3, "Matched Meal Type only"

        if d != "Any":
            lvl4 = df_in[df_in["Diet"].str.lower() == d.lower()]
            if not lvl4.empty:
                return lvl4, "Matched Diet only"

        return df_in, "No exact preference match found"

    df_temp, explanation_filter = get_filtered_df(df.copy(), cuisine, diet, meal)
    
    # Exclude cooked recently
    df_temp = df_temp[~df_temp["RecipeName"].isin(cooked_recently)]
    
    df_temp = compute_scores(df_temp, inventory_items, inventory_dict, inventory_text)

    # Use offset for pagination logic - Only return 1 for cleaner dashboard
    top_recipes = df_temp.sort_values("Final_Score", ascending=False).iloc[offset : offset + 1]
    results = []

    for _, r in top_recipes.iterrows():
        coverage_pct = round(r['Coverage'] * 100)
        semantic_pct = round(r['Semantic_Score'] * 100)
        
        explanation = (
            f"Strong recommendation with {coverage_pct}% ingredient coverage. "
            f"We found {len(r['Matched'])} item{'s' if len(r['Matched']) != 1 else ''} in your kitchen. "
            f"The recipe has a {semantic_pct}% semantic match with your stock. "
            f"{explanation_filter}."
        )

        ingredients_list = list(r["Cleaned_Ingredients"])
        results.append({
            "name": r["RecipeName"],
            "cuisine": r["Cuisine_Clean"],
            "meal": r["MealType"],
            "diet": r["Diet"],
            "semantic_score": round(r["Semantic_Score"], 2),
            "coverage_score": r["Coverage"],
            "expiry_score": r["Expiry"],
            "final_score": round(r["Final_Score"] * 100, 2),
            "matched": list(r["Matched"]) if not isinstance(r["Matched"], str) else [r["Matched"]],
            "missed": list(r["Missed"]) if not isinstance(r["Missed"], str) else [r["Missed"]],
            "instructions": r["TranslatedInstructions"],
            "explanation": explanation
        })
    return results

# -----------------------------
# Research-Level Evaluation
# -----------------------------
def evaluate_recommendation_system(scores_df, k=50):
    """
    Comprehensive evaluation of the recommendation system.
    Includes Ranking + Classification metrics.
    """
    from sklearn.metrics import (
        ndcg_score, confusion_matrix,
        precision_score, recall_score, f1_score
    )

    df_sorted = scores_df.sort_values("Final_Score", ascending=False)

    # Ground Truth: Recipe is relevant if cookable OR prevents waste
    y_true = (
        (df_sorted["Coverage"] >= 0.25) |
        (df_sorted["Expiry"] >= 0.30)
    ).astype(int).values

    # Ranking scores
    y_scores = df_sorted["Final_Score"].values

    # ===============================
    # 🧠 RANKING METRICS
    # ===============================

    # NDCG@8
    ndcg_8 = ndcg_score([y_true], [y_scores], k=8) if len(y_true) >= 8 else ndcg_score([y_true], [y_scores])

    # Precision@K
    precision_k = np.sum(y_true[:k]) / k if k else 0

    # Recall@K
    total_relevant = np.sum(y_true)
    recall_k = np.sum(y_true[:k]) / total_relevant if total_relevant else 0

    # F1@K
    f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) else 0

    # MRR
    mrr = 0
    for idx, rel in enumerate(y_true, start=1):
        if rel == 1:
            mrr = 1 / idx
            break

    # NDCG@K
    ndcg_k = ndcg_score([y_true], [y_scores], k=k)

    # ===============================
    # 🌱 SUSTAINABILITY METRICS
    # ===============================

    matchable = len(scores_df[scores_df["Final_Score"] >= 0.40])
    catalog_cov = matchable / len(scores_df) if len(scores_df) > 0 else 0

    expiring_used = 0
    total_expiring = 0
    for _, row in scores_df.iterrows():
        matched_count = len(row["Matched"])
        expiry_val = row["Expiry"]
        if expiry_val > 0:
            expiring_used += matched_count * expiry_val
            total_expiring += matched_count
    fwri = expiring_used / total_expiring if total_expiring else 0

    # ===============================
    # 🤖 CLASSIFICATION METRICS
    # ===============================

    y_pred = (df_sorted["Final_Score"] >= 0.30).astype(int).values

    cm = confusion_matrix(y_true, y_pred)
    # Extract TN, FP, FN, TP safely
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle cases where only one class is present in y_true or y_pred
        tn = cm[0, 0] if y_true.sum() == 0 and y_pred.sum() == 0 else 0
        tp = cm[0, 0] if y_true.sum() == len(y_true) and y_pred.sum() == len(y_pred) else 0
        fp = 0
        fn = 0

    precision_cls = precision_score(y_true, y_pred, zero_division=0)
    recall_cls = recall_score(y_true, y_pred, zero_division=0)
    f1_cls = f1_score(y_true, y_pred, zero_division=0)

    # ===============================
    # 📦 RETURN RESULTS
    # ===============================

    return {
        "ndcg_8": round(float(ndcg_8), 3),
        "f1_overall": round(float(f1_cls), 3), # Requested as F1 at the top

        # Ranking Metrics
        "precision_k": round(float(precision_k), 3),
        "recall_k": round(float(recall_k), 3),
        "f1_k": round(float(f1_k), 3),
        "mrr": round(float(mrr), 3),
        "ndcg_k": round(float(ndcg_k), 3),

        # Classification Metrics
        "precision": round(float(precision_cls), 3),
        "recall": round(float(recall_cls), 3),
        "f1": round(float(f1_cls), 3),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),

        # Sustainability
        "catalog_coverage": round(float(catalog_cov), 3),
        "fwri": round(float(fwri), 3),

        "k": k
    }

# -----------------------------
# PDF Generation
# -----------------------------

def generate_recipe_pdf(recipe_name, matched_ingredients, missed_ingredients, instructions):
    if not os.path.exists("temp_docs"):
        os.makedirs("temp_docs")
        
    pdf_filename = f"temp_docs/{recipe_name.replace(' ', '_')}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"Recipe: {recipe_name}", styles["Heading1"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Ingredients from your Kitchen:", styles["Heading2"]))
    if matched_ingredients:
        items = matched_ingredients.split(",") if isinstance(matched_ingredients, str) else matched_ingredients
        elements.append(ListFlowable([ListItem(Paragraph(i.strip(), styles["Normal"])) for i in items if i.strip()], bulletType='bullet'))
    else:
        elements.append(Paragraph("None", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    if missed_ingredients:
        elements.append(Paragraph("Items to Buy:", styles["Heading2"]))
        items = missed_ingredients.split(",") if isinstance(missed_ingredients, str) else missed_ingredients
        elements.append(ListFlowable([ListItem(Paragraph(i.strip(), styles["Normal"])) for i in items if i.strip()], bulletType='bullet'))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Instructions:", styles["Heading2"]))
    for line in instructions.split('\n'):
        if line.strip():
            elements.append(Paragraph(line.strip(), styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)
    return pdf_filename

# -----------------------------
# System Metrics
# -----------------------------

def calculate_system_metrics(user_inventory):
    """
    Calculates various metrics for the recommendation system based on user inventory.
    """
    if "Cuisine_Clean" not in df.columns:
        return None

    inventory_items = user_inventory
    inventory_dict = {item: 30 for item in user_inventory}
    inventory_text = " ".join(user_inventory)

    # Use the compute_scores function with ALL required 4 arguments
    scores_df = compute_scores(df.copy(), inventory_items, inventory_dict, inventory_text)
    
    matchable_count = len(scores_df[scores_df["Final_Score"] > 0.4])
    total_count = len(df)
    coverage = (matchable_count / total_count) * 100 if total_count > 0 else 0
    cuisine_counts = df["Cuisine_Clean"].value_counts().to_dict()
    
    dist = {
        "Ready to Cook (Score > 0.8)": len(scores_df[scores_df["Final_Score"] >= 0.8]),
        "Almost Ready (0.6 - 0.79)": len(scores_df[(scores_df["Final_Score"] >= 0.6) & (scores_df["Final_Score"] < 0.8)]),
        "Needs Shopping (< 0.6)": len(scores_df[scores_df["Final_Score"] < 0.6])
    }
    avg_score = scores_df["Final_Score"].mean() * 100

    return {
        "coverage": round(coverage, 2),
        "total_recipes": total_count,
        "matchable_recipes": matchable_count,
        "cuisine_distribution": cuisine_counts,
        "score_distribution": dist,
        "avg_match_score": round(avg_score, 2)
    }        
