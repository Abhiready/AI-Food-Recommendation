# data_processing.py (Simplified to remove dish logic)

import pandas as pd
import numpy as np

print("--- Starting Data Preprocessing ---")

# 1. Load Datasets
try:
    df_meta = pd.read_csv('Restaurant names and Metadata.csv')
    df_reviews = pd.read_csv('Restaurant reviews.csv', encoding='latin-1')
    print("✅ Datasets loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Make sure dataset files are in the directory.")
    exit()

# 2. Process Reviews Data
df_reviews.rename(columns={'Restaurant': 'Name'}, inplace=True)
df_reviews['Rating_Num'] = pd.to_numeric(df_reviews['Rating'], errors='coerce')
df_reviews.dropna(subset=['Rating_Num'], inplace=True)
restaurant_ratings = df_reviews.groupby('Name').agg(
    avg_rating=('Rating_Num', 'mean'),
    review_count=('Rating_Num', 'size')
).reset_index()
restaurant_ratings['avg_rating'] = round(restaurant_ratings['avg_rating'], 2)
print("✅ Reviews data processed.")

# 3. Process Metadata Data
df_meta['Cost'] = df_meta['Cost'].str.replace(',', '').astype(float)
df_meta['Collections'].fillna('', inplace=True)
print("✅ Metadata processed.")

# 4. Merge Datasets
df_final = pd.merge(df_meta, restaurant_ratings, on='Name', how='left')
df_final.dropna(subset=['avg_rating'], inplace=True)
print("✅ Datasets merged.")

# 5. Feature Engineering
# Simplified 'tags' feature without dishes
df_final['tags'] = df_final['Cuisines'] + ' ' + df_final['Collections']
print("✅ 'tags' feature created from Cuisines and Collections.")

# 6. Save Final Cleaned Data
output_columns = [
    'Name', 'Cuisines', 'Cost', 'avg_rating', 'review_count', 'tags',
    'Collections', 'Timings', 'Links'
]
df_final = df_final[output_columns]
df_final.to_csv('restaurants_cleaned.csv', index=False)

print("\n--- Data Preprocessing Complete ---")
print("🎉 Final cleaned data saved to 'restaurants_cleaned.csv'")

