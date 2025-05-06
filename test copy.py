import pandas as pd

# Assuming df is your original large DataFrame with language and genre columns
# Step 1: Identify the 5 most frequent languages
top_5_languages = df['language'].value_counts().nlargest(5).index.tolist()

# Step 2: Identify the 5 most frequent genres
top_5_genres = df['genre'].value_counts().nlargest(5).index.tolist()

# Step 3: Filter DataFrame to only include rows with top 5 languages AND top 5 genres
filtered_df = df[(df['language'].isin(top_5_languages)) & (df['genre'].isin(top_5_genres))]

# Step 4: Select 10,000 rows from the filtered DataFrame
if len(filtered_df) > 10000:
    # Randomly select 10,000 rows
    result_df = filtered_df.sample(n=10000, random_state=42)
else:
    # If fewer than 10,000 rows match the criteria, take all available rows
    result_df = filtered_df
    print(f"Warning: Only {len(filtered_df)} rows available with the top 5 languages and genres.")

# Verify the results
print(f"Languages in subset: {result_df['language'].unique()}")
print(f"Genres in subset: {result_df['genre'].unique()}")
print(f"Total rows selected: {len(result_df)}")