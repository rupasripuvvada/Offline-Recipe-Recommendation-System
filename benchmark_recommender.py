
import time
from recommender import recommend_recipes, load_inventory

inventory_data = load_inventory()

print("Measuring time for recommend_recipes('Any', 'Any', 'Any')...")
start_time = time.time()
recs = recommend_recipes("Any", "Any", "Any", preloaded_inventory=inventory_data)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
print(f"Number of recommendations: {len(recs)}")
if recs:
    print(f"Top recipe: {recs[0]['name']}")

print("\nMeasuring second run (should be faster due to caching)...")
start_time = time.time()
recs = recommend_recipes("Any", "Any", "Any", preloaded_inventory=inventory_data)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.4f} seconds")
