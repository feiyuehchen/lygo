import os
import pandas as pd
from utils import *

if __name__ == "__main__":
    processed_dir = '../dataset/processed'
    output_path = "../dataset/10k/score.csv"
    
    df = load_processed_data(processed_dir)

    # Get top 5 languages and tags
    top_3_languages = df['language'].value_counts().nlargest(3).index.tolist()
    # top_3_genres = df['tag'].value_counts().nlargest(3).index.tolist()
    top_3_genres = ['pop', 'rock', 'rap']

    # Filter dataframe to only include top languages and genres
    filtered_df = df[(df['language'].isin(top_3_languages)) & (df['tag'].isin(top_3_genres))]
    
    # Create balanced dataset
    result_df = pd.DataFrame()
    samples_per_combination = 100  # 10000 / (5*5)
    
    for language in top_3_languages:
        for tag in top_3_genres:
            combination_df = filtered_df[(filtered_df['language'] == language) & (filtered_df['tag'] == tag)]
            
            if len(combination_df) >= samples_per_combination:
                sampled = combination_df.sample(n=samples_per_combination, random_state=42)
            else:
                print(f"Warning: Only {len(combination_df)} samples available for language={language}, tag={tag}. Taking all available.")
                sampled = combination_df
            
            result_df = pd.concat([result_df, sampled])
    
    # Check if we have 10,000 samples or fewer
    total_samples = len(result_df)
    print(f"Total samples in balanced dataset: {total_samples}")
    
    # Save the balanced dataset
    result_df.to_csv(output_path, index=False)


    