from flask import Flask, render_template, request, redirect, session
import sqlite3
import os
import cv2
import pandas as pd
import re
import threading
import time
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
from recommender import recommend_recipes

app = Flask(__name__)
app.secret_key = "kitchen_secret"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# LOAD SHELF LIFE DATASET
# ---------------------------
shelf_df = pd.read_csv("shelf_life_data.csv")
shelf_df["Ingredient"] = shelf_df["Ingredient"].str.lower().str.strip()

# ---------------------------
# OCR
# ---------------------------
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# ---------------------------
# DATABASE INIT
# ---------------------------
def init_db():
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS grocery_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        serial_no INTEGER,
        item TEXT,
        quantity TEXT,
        purchase_date DATE,
        expiry_date DATE,
        days_left INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS inventory(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT,
        quantity TEXT,
        purchase_date DATE,
        expiry_date DATE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS grocery_list(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT,
        quantity TEXT,
        added_date DATE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cooked_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipe_name TEXT NOT NULL,
        cooked_date DATE NOT NULL,
        user_id TEXT
    )
    """)

    # Data Migration: Move from inventory to grocery_items if empty
    cur.execute("SELECT COUNT(*) FROM grocery_items")
    if cur.fetchone()[0] == 0:
        cur.execute("SELECT COUNT(*) FROM inventory")
        if cur.fetchone()[0] > 0:
            cur.execute("SELECT item, quantity, purchase_date, expiry_date FROM inventory")
            legacy_items = cur.fetchall()
            sno = 1
            for item, qty, p_date, e_date in legacy_items:
                try:
                    p_date_dt = datetime.strptime(p_date, '%Y-%m-%d').date() if p_date else datetime.now().date()
                    e_date_dt = datetime.strptime(e_date, '%Y-%m-%d').date() if e_date else (p_date_dt + timedelta(days=7))
                    days_left = (e_date_dt - datetime.now().date()).days
                except:
                    days_left = 7

                cur.execute("""
                INSERT INTO grocery_items (serial_no, item, quantity, purchase_date, expiry_date, days_left)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (sno, item, qty, p_date, e_date, days_left))
                sno += 1

    conn.commit()
    conn.close()

init_db()

# ---------------------------
# AUTO-DELETE EXPIRED ITEMS
# ---------------------------
def auto_cleanup_expired():
    """Background task: deletes items with days_left < 0 from grocery_items daily."""
    while True:
        try:
            conn = sqlite3.connect("grocery_ocr.db")
            cur = conn.cursor()
            cur.execute("DELETE FROM grocery_items WHERE days_left < 0")
            deleted = cur.rowcount
            conn.commit()
            conn.close()
            if deleted > 0:
                print(f"[Auto-Cleanup] Removed {deleted} expired item{'s' if deleted != 1 else ''} from inventory.")
        except Exception as e:
            print(f"[Auto-Cleanup] Error: {e}")
        time.sleep(86400)  # Run once every 24 hours

cleanup_thread = threading.Thread(target=auto_cleanup_expired, daemon=True)
cleanup_thread.start()

# ---------------------------
# SHELF LIFE CALCULATION
# ---------------------------
def calculate_expiry(item):
    today = datetime.today().date()
    row = shelf_df[shelf_df["Ingredient"].str.contains(item.lower(), na=False)]

    if not row.empty:
        fridge = int(row["Refrigerator_Shelf_Life_Days"].values[0])
        pantry = int(row["Pantry_Shelf_Life_Days"].values[0])
        avg_days = round((0.7 * fridge) + (0.3 * pantry))
    else:
        avg_days = 7

    expiry = today + timedelta(days=avg_days)
    days_left = (expiry - today).days

    return str(today), str(expiry), int(days_left)

# ---------------------------
# QUANTITY PARSING UTILS
# ---------------------------
def parse_qty(q):
    match = re.search(r"(\d+\.?\d*)", str(q))
    return float(match.group(1)) if match else 0.0

def get_unit(q):
    unit_patterns = ["kg", "g", "gm", "gram", "grams", "ltr", "liter", "lit", "ml",
                     "pcs", "piece", "packet", "pack", "box", "bunch", "jar", "units", "unit", "spoon", "spoons"]
    q_lower = str(q).lower()
    for u in unit_patterns:
        if re.search(r'\b' + u + r'\b', q_lower):
            return u
    return "unit"

def convert_to_base(val, unit):
    """Normalize quantities to base units (g, ml, pcs) for math."""
    unit = unit.lower()
    if unit in ["kg", "kgs"]:
        return val * 1000, "g"
    if unit in ["ltr", "liter", "liters", "l", "lit"]:
        return val * 1000, "ml"
    if unit in ["spoon", "spoons"]:
        return val * 2, "g"
    if unit in ["gm", "gram", "grams"]:
        return val, "g"
    return val, unit

# ---------------------------
# LOGIN
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_id = request.form["login_id"]
        password = request.form["password"]

        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE (email=? OR username=?) AND password=?",
            (login_id, login_id, password)
        )
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = user[1]
            return redirect("/home")
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------------------
# FORGOT PASSWORD
# ---------------------------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not new_password:
            return render_template("forgot_password.html", error="All fields are required.")

        if new_password != confirm_password:
            return render_template("forgot_password.html", error="Passwords do not match.")

        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username=?", (username,))
        user = cur.fetchone()

        if not user:
            conn.close()
            return render_template("forgot_password.html", error="Username not found.")

        cur.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
        conn.commit()
        conn.close()
        return render_template("forgot_password.html", success="Password reset successful!")

    return render_template("forgot_password.html")

# ---------------------------
# REGISTER
# ---------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users(username,email,password) VALUES(?,?,?)",
                (username, email, password)
            )
            conn.commit()
            session["user"] = username
            conn.close()
            return redirect("/home")
        except Exception as e:
            conn.close()
            return render_template("register.html", error="Registration failed")

    return render_template("register.html")

# ---------------------------
# HOME
# ---------------------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    plan = session.get("meal_plan", {})
    return render_template("home.html", plan=plan)

@app.route("/meal_planner")
def meal_planner():
    if "user" not in session:
        return redirect("/")

    from recommender import recommend_recipes

    if "meal_plan" not in session:
        session["meal_plan"] = {
            "Breakfast": {"index": 0},
            "Lunch": {"index": 0},
            "Dinner": {"index": 0}
        }

    plan = session["meal_plan"]
    grocery_list = set()

    five_days_ago = (datetime.now() - timedelta(days=5)).date()
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT recipe_name FROM cooked_history WHERE cooked_date >= ?", (five_days_ago,))
    cooked_recently = {row[0] for row in cur.fetchall()}
    conn.close()

    detailed_plan = {}
    for meal_type in ["Breakfast", "Lunch", "Dinner"]:
        recs = recommend_recipes(cuisine="Any", diet="Any", meal=meal_type)
        recs = [r for r in recs if r["name"] not in cooked_recently]

        if recs:
            idx = plan[meal_type].get("index", 0) % len(recs)
            recipe = recs[idx]
            detailed_plan[meal_type] = recipe
            if recipe.get("missed"):
                items = [i.strip() for i in recipe["missed"]] if isinstance(recipe["missed"], list) else [i.strip() for i in recipe["missed"].split(",") if i.strip()]
                grocery_list.update(items)
        else:
            detailed_plan[meal_type] = None

    return render_template("meal_planner.html", plan=detailed_plan, grocery=sorted(list(grocery_list)))

@app.route("/meal_planner/refresh/<meal_type>", methods=["GET", "POST"])
def refresh_meal(meal_type):
    if "user" not in session:
        return redirect("/")

    if "meal_plan" not in session:
        session["meal_plan"] = {
            "Breakfast": {"index": 0},
            "Lunch": {"index": 0},
            "Dinner": {"index": 0}
        }

    plan = session["meal_plan"]
    if meal_type in plan:
        plan[meal_type]["index"] = plan[meal_type].get("index", 0) + 1
        session.modified = True

    return redirect("/meal_planner")

@app.route("/meal_planner/cooked/<recipe_name>", methods=["GET", "POST"])
def mark_as_cooked(recipe_name):
    if "user" not in session:
        return redirect("/")
    return redirect(f"/cooked_update/{recipe_name}")

@app.route("/cooked_update/<recipe_name>")
def cooked_update(recipe_name):
    if "user" not in session:
        return redirect("/")

    from recommender import df
    recipe_row = df[df["RecipeName"] == recipe_name]
    if recipe_row.empty:
        return "Recipe not found", 404

    ingredients = recipe_row.iloc[0]["Cleaned_Ingredients"]

    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT item, quantity FROM grocery_items")
    inventory = cur.fetchall()
    conn.close()

    matched_ingredients = []
    for ing in ingredients:
        matches = []
        ing_clean = ing.lower().strip()
        for item, qty in inventory:
            item_clean = item.lower().strip()
            # Use flexible word boundaries to allow singular/plural matches (e.g. Pineapple -> Pineapples)
            # but prevent false positives (e.g. Ice -> Rice)
            pattern = r'\b' + re.escape(ing_clean) + r'(s|es)?\b'
            rev_pattern = r'\b' + re.escape(item_clean) + r'(s|es)?\b'
            
            if re.search(pattern, item_clean, re.IGNORECASE) or \
               re.search(rev_pattern, ing_clean, re.IGNORECASE):
                matches.append({"item": item, "quantity": qty})

        matched_ingredients.append({
            "name": ing,
            "matches": matches
        })

    return render_template("cooked_update.html", recipe_name=recipe_name, ingredients=matched_ingredients)

@app.route("/process_cooked_update", methods=["POST"])
def process_cooked_update():
    if "user" not in session:
        return redirect("/")

    recipe_name = request.form.get("recipe_name")
    items_to_update = request.form.getlist("items[]")
    quantities = request.form.getlist("quantities[]")

    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()

    for item, used_qty_str in zip(items_to_update, quantities):
        if used_qty_str and item:
            cur.execute("SELECT quantity FROM grocery_items WHERE item = ?", (item,))
            row = cur.fetchone()
            if row:
                current_qty_str = row[0]
                current_val = parse_qty(current_qty_str)
                used_val = parse_qty(used_qty_str)
                current_unit = get_unit(current_qty_str)
                used_unit = get_unit(used_qty_str)

                # Convert both to base units for calculation
                base_current_val, base_current_unit = convert_to_base(current_val, current_unit)
                base_used_val, base_used_unit = convert_to_base(used_val, used_unit)

                # Only subtract if base units are compatible (e.g., both are weight or both are volume)
                if base_current_unit == base_used_unit:
                    new_val_base = base_current_val - base_used_val
                    
                    if new_val_base <= 0:
                        cur.execute("DELETE FROM grocery_items WHERE item = ?", (item,))
                    else:
                        # If we started with kg and result is still large, stay in g or convert back?
                        # Let's keep the base unit for better precision after partial use.
                        formatted_val = int(new_val_base) if new_val_base == int(new_val_base) else round(new_val_base, 2)
                        new_qty_str = f"{formatted_val} {base_current_unit}"
                        cur.execute("UPDATE grocery_items SET quantity = ? WHERE item = ?", (new_qty_str, item))
                else:
                    # Units mismatch and not convertible (e.g., kg vs ml), keep as is or log error?
                    # For now, silently skip or we could just overwrite with new_qty_str if we wanted.
                    pass

    today = datetime.now().date()
    cur.execute("INSERT INTO cooked_history (recipe_name, cooked_date, user_id) VALUES (?, ?, ?)",
                (recipe_name, today, session.get("user", "default")))

    conn.commit()
    conn.close()

    return redirect("/meal_planner")

@app.route("/metrics")
def metrics():
    if "user" not in session:
        return redirect("/")

    from recommender import calculate_system_metrics

    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT item FROM grocery_items")
    inventory_items = [row[0] for row in cur.fetchall()]
    conn.close()

    stats = calculate_system_metrics(inventory_items)
    return render_template("metrics.html", stats=stats)

@app.route("/research_metrics")
def research_metrics():
    if "user" not in session:
        return redirect("/")

    from recommender import compute_scores, evaluate_recommendation_system, df, load_inventory

    inventory_items, inventory_dict, inventory_text = load_inventory()

    if not inventory_items:
        return render_template("research_metrics.html", metrics=None)

    scores_df = compute_scores(df.copy(), inventory_items, inventory_dict, inventory_text)
    metrics = evaluate_recommendation_system(scores_df)

    return render_template("research_metrics.html", metrics=metrics)

# ---------------------------
# MODEL COMPARISON
# ---------------------------
@app.route("/model_comparison")
def model_comparison():
    if "user" not in session:
        return redirect("/")

    import json

    metrics_path = os.path.join("static", "comparison_results.json")
    if not os.path.exists(metrics_path):
        return render_template("model_comparison.html", error="Comparison metrics not generated yet. Please run compare_models.py.")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return render_template("model_comparison.html", baseline=data["baseline"], proposed=data["proposed"])


# ---------------------------
# OCR ITEM EXTRACTION (IMPROVED)
# ---------------------------

# Words/phrases to skip entirely — expanded and lowercased
SKIP_WORDS = {
    "subtotal", "total", "delivery", "charge", "order", "summary", "price","Fotal",
    "track", "www", "grand", "paid", "date", "orderid", "id", "jm", "jiomart",
    "items", "bial", "drage", "dmart", "vegetables", "greens", "meat", "seafood",
    "diary", "refrigerated", "grains", "pulses", "pluses", "flours", "oils",
    "liquids", "spices", "seasonings", "idli", "dosa", "essentials", "herbs",
    "other", "ingredients", "snacks", "chutney", "discount", "offer", "qty",
    "mrp", "amount", "rs", "inr", "tax", "gst", "hsn", "invoice", "bill",
    "receipt", "store", "shop", "address", "phone", "email", "thank", "you",
    "visit", "again", "cash", "card", "upi", "payment", "net", "payable",
    "saving", "savings", "free", "off", "buy", "get","Fotal","Y","Grainspulsesflours Soojj"
}

# Known unit tokens (for detection and normalization)
UNIT_TOKENS = [
    "kg", "kgs", "g", "gm", "gms", "gram", "grams",
    "ltr", "ltrs", "liter", "liters", "litre", "litres",
    "ml", "pcs", "pc", "piece", "pieces",
    "packet", "packets", "pack", "packs",
    "box", "boxes", "bunch", "bunches",
    "jar", "jars", "bottle", "bottles",
    "dozen", "doz", "no", "nos", "unit", "units",
    "strip", "strips", "sachet", "sachets",
    "can", "cans", "pouch", "pouches"
]

# Normalize unit aliases to canonical form
UNIT_ALIASES = {
    "kgs": "kg", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "ltr": "L", "ltrs": "L", "liter": "L", "liters": "L", "litre": "L", "litres": "L",
    "ml": "ml", "pcs": "pcs", "pc": "pcs", "piece": "pcs", "pieces": "pcs",
    "packet": "pkt", "packets": "pkt", "pack": "pkt", "packs": "pkt",
    "box": "box", "boxes": "box", "bunch": "bunch", "bunches": "bunch",
    "jar": "jar", "jars": "jar", "bottle": "bottle", "bottles": "bottle",
    "dozen": "doz", "doz": "doz", "no": "pcs", "nos": "pcs",
    "unit": "pcs", "units": "pcs", "strip": "strip", "strips": "strip",
    "sachet": "sachet", "sachets": "sachet", "can": "can", "cans": "can",
    "pouch": "pouch", "pouches": "pouch"
}

# OCR text corrections for common misreads
OCR_TEXT_CORRECTIONS = {
    "bestroot": "beetroot", "bestroots": "beetroot",
    "avccado": "avocado", "avccados": "avocados",
    "tomotoes": "tomatoes", "tomotoe": "tomato",
    "tamato": "tomato", "tamatos": "tomatoes",
    "potatoe": "potato", "potatoes": "potato",
    "onoin": "onion", "onions": "onion",
    "garlc": "garlic", "garic": "garlic",
    "chillie": "chilli", "chillies": "chilli",
    "chili": "chilli", "chilies": "chilli",
    "corainder": "coriander", "corriander": "coriander",
    "cumin seed": "cumin", "cumin seeds": "cumin",
    "turmeric powder": "turmeric",
    "red chili powder": "red chilli", "red chilli powder": "red chilli",
    "coriander powder": "coriander",
    "supar": "sugar", "suggar": "sugar",
    "milklitres": "milk", "milkltrs": "milk",
    "breadbrown": "brown bread", "breadwhite": "white bread",
    "ketchupbottle": "ketchup",
    "pastapack": "pasta", "noodlespack": "noodles",
    "jambottle": "jam",
    "pani puris": "pani puri",
    "egg pcs": "eggs", "eggs pcs": "eggs","Mushroomse":"mushrooms"
}

def clean_item_name(raw_name):
    """
    Clean and normalize an OCR-extracted item name.
    Returns a properly capitalized, cleaned string or None if invalid.
    """
    # Remove non-alphabetic/space characters
    name = re.sub(r"[^a-zA-Z\s]", "", raw_name).strip().lower()

    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # Skip if too short or is a unit/skip word
    if len(name) < 2:
        return None

    # Check against skip words (word-level)
    words = name.split()
    if all(w in SKIP_WORDS for w in words):
        return None
    # Skip if any single word exactly matches a unit token
    if len(words) == 1 and name in UNIT_TOKENS:
        return None

    # Apply OCR corrections
    if name in OCR_TEXT_CORRECTIONS:
        name = OCR_TEXT_CORRECTIONS[name]

    # Title case the final name
    return name.strip().title()


def normalize_quantity(number_str, unit_str):
    """
    Given a raw number string and unit string, return a clean 'X unit' string.
    """
    try:
        num = float(number_str)
        # Format: if it's a whole number, drop decimal
        formatted_num = int(num) if num == int(num) else round(num, 2)
    except ValueError:
        formatted_num = 1

    unit = unit_str.lower().strip()
    canonical_unit = UNIT_ALIASES.get(unit, unit if unit in UNIT_TOKENS else "pcs")

    return f"{formatted_num} {canonical_unit}"


def extract_items(image_path):
    """
    Improved OCR extraction pipeline:
    1. Pre-process image for better OCR accuracy
    2. Extract all text lines with bounding box info
    3. Use regex-based pattern matching to find item+quantity pairs
    4. Merge duplicates and normalize
    """
    # ── Step 1: Image preprocessing ──────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold for better text/background separation
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Scale up small images for better OCR
    h, w = thresh.shape
    if w < 800:
        scale = 800 / w
        thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Save preprocessed image temporarily
    preprocessed_path = image_path + "_preprocessed.png"
    cv2.imwrite(preprocessed_path, thresh)

    # ── Step 2: OCR ──────────────────────────────────────────────────────────
    result = ocr.ocr(preprocessed_path)

    # Clean up temp file
    try:
        os.remove(preprocessed_path)
    except:
        pass

    if not result or not result[0]:
        return []

    # ── Step 3: Collect all text tokens with confidence scores ───────────────
    # Each OCR result: [bbox, (text, confidence)]
    lines = []
    for page in result:
        if page:
            for detection in page:
                text = detection[1][0].strip()
                confidence = detection[1][1]
                if text and confidence > 0.5:  # Filter low-confidence detections
                    lines.append(text)

    # Join lines into a single text block for pattern matching
    full_text = "\n".join(lines)

    # ── Step 4: Pattern-based extraction ─────────────────────────────────────
    # Pattern A: "Item Name  Qty Unit"  e.g. "Tomato  2 kg"
    # Pattern B: "Item Name  Qty"       e.g. "Onion  3"
    # Pattern C: "Qty Unit Item"        e.g. "2kg Potatoes"
    # Pattern D: Inline e.g.            "Milk 1ltr" or "Eggs 12pcs"

    units_pattern = "|".join(re.escape(u) for u in sorted(UNIT_TOKENS, key=len, reverse=True))

    # Patterns (applied to each line)
    # Match: ITEM_NAME  NUMBER UNIT  (or just NUMBER)
    pattern_item_qty = re.compile(
        r"^([a-zA-Z][a-zA-Z\s\-&'\.]{1,40}?)\s+"        # item name (2+ chars)
        r"(\d+(?:\.\d+)?)\s*"                              # number
        r"(" + units_pattern + r")?",                      # optional unit
        re.IGNORECASE
    )

    # Match: NUMBER UNIT  ITEM_NAME  (quantity-first format)
    pattern_qty_item = re.compile(
        r"^(\d+(?:\.\d+)?)\s*"                            # number
        r"(" + units_pattern + r")?\s+"                   # optional unit
        r"([a-zA-Z][a-zA-Z\s\-&'\.]{1,40})$",            # item name
        re.IGNORECASE
    )

    # Match inline: e.g. "Onion2kg", "Milk1ltr"
    pattern_inline = re.compile(
        r"([a-zA-Z][a-zA-Z\s\-&'\.]{1,30?})"             # item name
        r"(\d+(?:\.\d+)?)"                                 # number
        r"(" + units_pattern + r")?",                      # optional unit
        re.IGNORECASE
    )

    raw_items = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are clearly not item lines
        line_lower = line.lower()
        if any(skip in line_lower for skip in ["subtotal", "total", "grand total",
                                                "delivery", "www.", "orderid",
                                                "invoice", "receipt", "address",
                                                "phone", "email", "gst", "hsn",
                                                "payment", "cash", "upi", "payable"]):
            continue

        # Skip lines that are purely numeric (prices, dates, codes)
        if re.match(r"^[\d\s\.\-/\:₹$,]+$", line):
            continue

        # Try Pattern A: Item + Qty [Unit]
        match = pattern_item_qty.match(line)
        if match:
            item_raw = match.group(1).strip()
            number = match.group(2)
            unit = match.group(3) or ""
            item_clean = clean_item_name(item_raw)
            if item_clean:
                qty = normalize_quantity(number, unit) if unit else f"{number} pcs"
                raw_items.append((item_clean, qty))
                continue

        # Try Pattern B: Qty [Unit] + Item
        match = pattern_qty_item.match(line)
        if match:
            number = match.group(1)
            unit = match.group(2) or ""
            item_raw = match.group(3).strip()
            item_clean = clean_item_name(item_raw)
            if item_clean:
                qty = normalize_quantity(number, unit) if unit else f"{number} pcs"
                raw_items.append((item_clean, qty))
                continue

        # Try Pattern C: Inline e.g. "Onion2kg"
        match = pattern_inline.match(line)
        if match:
            item_raw = match.group(1).strip()
            number = match.group(2)
            unit = match.group(3) or ""
            item_clean = clean_item_name(item_raw)
            if item_clean:
                qty = normalize_quantity(number, unit) if unit else f"{number} pcs"
                raw_items.append((item_clean, qty))
                continue

    # ── Step 5: Fallback — sequential word-pair scan ──────────────────────────
    # If we got very few items from pattern matching, fall back to word-pair scan
    if len(raw_items) < 2:
        raw_items = _fallback_word_pair_scan(lines)

    # ── Step 6: Merge duplicates ──────────────────────────────────────────────
    merged = {}
    for item_name, qty_val in raw_items:
        # Canonicalize key
        key_name = item_name.lower().strip()

        # Apply OCR corrections on key
        if key_name in OCR_TEXT_CORRECTIONS:
            item_name = OCR_TEXT_CORRECTIONS[key_name].title()
            key_name = item_name.lower()

        num = parse_qty(qty_val)
        unit = get_unit(qty_val)
        key = (key_name, unit)

        if key not in merged:
            merged[key] = {"name": item_name, "total": 0.0, "unit": unit}
        merged[key]["total"] += num

    final_items = []
    for key, val in merged.items():
        total = val["total"]
        formatted = int(total) if total == int(total) else round(total, 2)
        unit = UNIT_ALIASES.get(val["unit"], val["unit"])
        final_items.append((val["name"], f"{formatted} {unit}"))

    return final_items


def _fallback_word_pair_scan(lines):
    """
    Fallback word-pair scan when pattern matching yields too few results.
    Collects consecutive alphabetic words into a multi-word item name, then
    reads the following number+unit as the quantity.  This prevents
    multi-word spice/ingredient names like "Cinnamon Stick" or "Cardamom Pod"
    from being stored as only their last word.
    """
    words = []
    for line in lines:
        for token in line.split():
            words.append(token.lower().strip())

    items = []
    i = 0
    while i < len(words):
        token = words[i]

        # Skip pure-numeric or empty tokens
        if not token or re.match(r"^[\d\.\-/\:]+$", token):
            i += 1
            continue

        # Skip skip-words and unit-only tokens at the start
        cleaned = re.sub(r"[^a-zA-Z\s]", "", token).strip()
        if not cleaned or cleaned in SKIP_WORDS or cleaned in UNIT_TOKENS:
            i += 1
            continue

        # ── Collect a multi-word name phrase ─────────────────────────────────
        # Keep accumulating words as long as they are alphabetic, not a unit
        # token, not a skip-word, and not a digit token.
        name_words = [cleaned]
        j = i + 1
        while j < len(words):
            nxt = re.sub(r"[^a-zA-Z\s]", "", words[j]).strip()
            # Stop if the next token contains digits (quantity incoming)
            if re.search(r"\d", words[j]):
                break
            # Stop if next word is a unit or skip word — it likely belongs to
            # the *next* item's quantity, not to this name.
            if nxt in UNIT_TOKENS or nxt in SKIP_WORDS or not nxt:
                break
            name_words.append(nxt)
            j += 1

        # ── Look for a quantity right after the collected name ────────────────
        if j < len(words) and re.search(r"\d", words[j]):
            qty_token = words[j]
            full_qty = qty_token
            # Peek ahead for a unit token
            if j + 1 < len(words) and words[j + 1] in UNIT_TOKENS:
                full_qty = f"{qty_token} {words[j + 1]}"
                j += 1  # will be skipped when we set i = j + 1

            # Normalize inline qty like "1kg" -> "1 kg"
            full_qty = re.sub(r"(\d+\.?\d*)\s*([a-zA-Z]+)", r"\1 \2", full_qty)

            item_phrase = " ".join(name_words)
            item_clean = clean_item_name(item_phrase)
            if item_clean:
                num_match = re.search(r"(\d+\.?\d*)", full_qty)
                unit_match = re.search(r"([a-zA-Z]+)$", full_qty.strip())
                number = num_match.group(1) if num_match else "1"
                unit = unit_match.group(1).lower() if unit_match else ""
                qty = normalize_quantity(number, unit) if unit else f"{number} pcs"
                items.append((item_clean, qty))
            i = j + 1
        else:
            # No quantity found — skip just the first word and retry
            i += 1

    return items


# ---------------------------
# UPLOAD BILL
# ---------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["bill"]

        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            items = extract_items(path)

            conn = sqlite3.connect("grocery_ocr.db")
            cur = conn.cursor()

            cur.execute("SELECT MAX(serial_no) FROM grocery_items")
            res = cur.fetchone()[0]
            sno = (res if res else 0) + 1

            for item, qty in items:
                purchase, expiry, days_left = calculate_expiry(item)
                cur.execute("""
                INSERT INTO grocery_items(serial_no, item, quantity, purchase_date, expiry_date, days_left)
                VALUES(?,?,?,?,?,?)
                """, (sno, item, qty, purchase, expiry, days_left))

                cur.execute("""
                INSERT INTO inventory(item, quantity, purchase_date, expiry_date)
                VALUES(?,?,?,?)
                """, (item, qty, purchase, expiry))
                sno += 1

            conn.commit()
            conn.close()

            return redirect("/inventory")

    return render_template("upload.html")

# ---------------------------
# MANUAL ADD
# ---------------------------
@app.route("/manual_add", methods=["POST"])
def manual_add():
    item = request.form["item"]
    qty = request.form["qty"]

    purchase, expiry, days_left = calculate_expiry(item)

    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()

    cur.execute("SELECT MAX(serial_no) FROM grocery_items")
    res = cur.fetchone()[0]
    sno = (res if res else 0) + 1

    cur.execute("""
    INSERT INTO grocery_items(serial_no, item, quantity, purchase_date, expiry_date, days_left)
    VALUES(?,?,?,?,?,?)
    """, (sno, item, qty, purchase, expiry, days_left))

    cur.execute("""
    INSERT INTO inventory(item, quantity, purchase_date, expiry_date)
    VALUES(?,?,?,?)
    """, (item, qty, purchase, expiry))

    conn.commit()
    conn.close()

    return redirect("/inventory")


# ---------------------------
# CONSOLIDATE INVENTORY
# ---------------------------
def consolidate_inventory():
    """
    Persistently merges duplicate items in grocery_items.
    """
    conn = sqlite3.connect("grocery_ocr.db")
    df = pd.read_sql("SELECT * FROM grocery_items", conn)

    if df.empty:
        conn.close()
        return

    ocr_corrections = {
        "ketchupbottle": "Ketchup",
        "bestroot": "Beetroot",
        "breadbrown": "Brown Bread",
        "breadwhite": "White Bread",
        "milklitres": "Milk",
        "egg pcs": "Eggs",
        "avccados": "Avocados",
        "pastapack": "Pasta",
        "noodlespack": "Noodles",
        "jambottle": "Jam",
        "supar": "Sugar",
        "pani puris": "Pani Puri",
        "cumin seeds": "Cumin",
        "turmeric powder": "Turmeric",
        "red chili powder": "Red Chilli",
        "red chilli powder": "Red Chilli",
        "coriander powder": "Coriander",
    }

    def canonical(name):
        name = str(name).lower().strip()
        if name in ocr_corrections:
            return ocr_corrections[name].lower()
        words = name.split()
        deduped = [words[0]] if words else []
        for w in words[1:]:
            if w != deduped[-1]:
                deduped.append(w)
        name = " ".join(deduped)
        name = re.sub(r"oes$", "o", name)
        name = re.sub(r"(?<=[^aeiou])s$", "", name)
        return name.strip()

    df["expiry_date"] = pd.to_datetime(df["expiry_date"], format="mixed", errors="coerce")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["expiry_date", "purchase_date"])

    df["canonical"] = df["item"].apply(canonical)
    to_delete = []

    for canon_key, group in df.groupby("canonical"):
        primary_id = group["id"].iloc[0]
        total_qty = 0.0
        units = []
        for q in group["quantity"]:
            total_qty += parse_qty(q)
            units.append(get_unit(q))

        unit = max(set(units), key=units.count) if units else "unit"
        qty_num = int(total_qty) if total_qty == int(total_qty) else total_qty
        new_qty = f"{qty_num} {unit}"

        earliest_purchase = group["purchase_date"].min()
        latest_expiry = group["expiry_date"].max()
        days_left = (latest_expiry - pd.Timestamp.now().normalize()).days

        clean_name = ocr_corrections.get(canon_key, canon_key.title())

        conn.execute(
            "UPDATE grocery_items SET item=?, quantity=?, purchase_date=?, expiry_date=?, days_left=? WHERE id=?",
            (clean_name, new_qty, str(earliest_purchase.date()),
             str(latest_expiry.date()), int(days_left), int(primary_id))
        )
        to_delete.extend(group["id"].iloc[1:].tolist())

    if to_delete:
        conn.execute(
            f"DELETE FROM grocery_items WHERE id IN ({','.join(['?'] * len(to_delete))})",
            tuple(to_delete)
        )

    conn.commit()
    conn.close()


# ---------------------------
# INVENTORY DASHBOARD
# ---------------------------
@app.route("/inventory")
def inventory():
    if "user" not in session:
        return redirect("/")

    consolidate_inventory()

    conn = sqlite3.connect("grocery_ocr.db")
    df = pd.read_sql("SELECT * FROM grocery_items", conn)
    conn.close()

    if df.empty:
        return render_template("inventory.html", items=[], summary={}, chart_data={})

    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    df = df.dropna(subset=["expiry_date"])
    today = pd.Timestamp(datetime.today().date())
    df["days_left"] = (df["expiry_date"] - today).dt.days

    def get_status(days):
        if days < 0:
            return "Expired"
        elif days <= 3:
            return "Urgent"
        elif days <= 7:
            return "Use Soon"
        else:
            return "Fresh"

    df["status"] = df["days_left"].apply(get_status)

    color_map = {
        "Fresh": "success",
        "Use Soon": "warning",
        "Urgent": "danger",
        "Expired": "dark"
    }

    df["color"] = df["status"].map(color_map)
    df = df.sort_values(by="days_left")

    summary = {
        "total": len(df),
        "fresh": len(df[df.status == "Fresh"]),
        "urgent": len(df[df.status == "Urgent"]),
        "wasted": len(df[df.status == "Expired"])
    }

    chart_data = {
        "Fresh": summary["fresh"],
        "Urgent": summary["urgent"],
        "Expired": summary["wasted"]
    }

    items = df.to_dict(orient="records")

    return render_template(
        "inventory.html",
        items=items,
        summary=summary,
        chart_data=chart_data
    )


# ---------------------------
# UPDATE QUANTITY
# ---------------------------
@app.route("/update_quantity/<int:id>", methods=["POST"])
def update_quantity(id):
    quantity = request.form["quantity"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("UPDATE grocery_items SET quantity=? WHERE id=?", (quantity, id))
    conn.commit()
    conn.close()
    return redirect("/inventory")

@app.route("/mark_used/<int:id>", methods=["POST"])
def mark_used(id):
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_items WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect("/inventory")

@app.route("/inventory/clear", methods=["POST"])
def clear_inventory():
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_items")
    conn.commit()
    conn.close()
    return redirect("/inventory")

@app.route("/mark_wasted/<int:id>", methods=["POST"])
def mark_wasted(id):
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_items WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect("/inventory")


# ---------------------------
# GROCERY LIST
# ---------------------------
@app.route("/grocery_list")
def grocery_list():
    if "user" not in session:
        return redirect("/")

    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM grocery_list ORDER BY added_date DESC")
    items = cur.fetchall()
    conn.close()

    formatted_items = [
        {"id": item[0], "item": item[1], "quantity": item[2], "added_date": item[3]}
        for item in items
    ]

    return render_template("grocery_list.html", items=formatted_items)

@app.route("/grocery_list/add", methods=["POST"])
def add_grocery_item():
    item = request.form.get("item")
    qty = request.form.get("qty")
    today = datetime.today().date()

    if item:
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO grocery_list(item, quantity, added_date) VALUES(?,?,?)", (item, qty, today))
        conn.commit()
        conn.close()

    return redirect("/grocery_list")

@app.route("/grocery_list/add_missing", methods=["POST"])
def add_missing_ingredients():
    missing_str = request.form.get("missing", "")
    if missing_str:
        items = [i.strip() for i in missing_str.split(",") if i.strip()]
        return render_template("confirm_grocery.html", items=items)
    return redirect("/grocery_list")

@app.route("/grocery_list/finalize_bulk_add", methods=["POST"])
def finalize_bulk_add():
    items = request.form.getlist("items[]")
    quantities = request.form.getlist("quantities[]")
    today = datetime.today().date()

    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    for item, qty in zip(items, quantities):
        if item and qty:
            cur.execute("INSERT INTO grocery_list(item, quantity, added_date) VALUES(?,?,?)", (item, qty, today))
    conn.commit()
    conn.close()

    return redirect("/grocery_list")

@app.route("/grocery_list/delete/<int:id>", methods=["POST"])
def delete_grocery_item(id):
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_list WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect("/grocery_list")

@app.route("/grocery_list/clear", methods=["POST"])
def clear_grocery_list():
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_list")
    conn.commit()
    conn.close()
    return redirect("/grocery_list")


# ---------------------------
# CONTEXT PROCESSOR
# ---------------------------
@app.context_processor
def inject_notifications():
    if "user" not in session:
        return {"expiry_notifs": []}

    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT item, days_left FROM grocery_items WHERE days_left <= 3 AND days_left >= 0 ORDER BY days_left ASC LIMIT 5"
    )
    rows = cur.fetchall()
    conn.close()
    return {"expiry_notifs": rows}


# ---------------------------
# RECIPE RECOMMENDATION
# ---------------------------
@app.route("/recipes", methods=["GET", "POST"])
def recipes():
    recipes_list = []
    cuisine = "Any"
    diet = "Any"
    meal = "Any"
    offset = request.args.get("offset", 0, type=int)

    if request.method == "POST":
        cuisine = request.form.get("cuisine", "Any")
        diet = request.form.get("diet", "Any")
        meal = request.form.get("meal", "Any")
        offset = 0
    elif request.args.get("cuisine"):
        cuisine = request.args.get("cuisine", "Any")
        diet = request.args.get("diet", "Any")
        meal = request.args.get("meal", "Any")

    recipes_list = recommend_recipes(cuisine, diet, meal, offset=offset)

    return render_template(
        "recipes.html",
        recipes=recipes_list,
        cuisine=cuisine,
        diet=diet,
        meal=meal,
        offset=offset
    )

@app.route("/download_pdf/<recipe_name>")
def download_pdf(recipe_name):
    if "user" not in session:
        return redirect("/")

    from recommender import generate_recipe_pdf

    df_csv = pd.read_csv("Final_Cleaned_Dataset.csv")
    recipe_row = df_csv[df_csv["RecipeName"] == recipe_name]

    if recipe_row.empty:
        return "Recipe not found", 404

    row = recipe_row.iloc[0]
    ingredients = ", ".join(eval(row["Cleaned_Ingredients"]))
    instructions = row["TranslatedInstructions"]

    pdf_path = generate_recipe_pdf(recipe_name, ingredients, "", instructions)

    from flask import send_file
    return send_file(pdf_path, as_attachment=True)


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
