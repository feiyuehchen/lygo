import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from collections import defaultdict
import glob
from utils import *








def filter_songs_by_criteria(df, language=None, genre=None, year_start=None, year_end=None):
    """
    Filter songs based on specified criteria
    """
    filtered_df = df.copy()
    
    # Filter by language
    if language:
        filtered_df = filtered_df[filtered_df['language'] == language]
    
    # Filter by genre
    if genre:
        if 'tag' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['tag'] == genre]
        elif 'genre' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['genre'] == genre]
    
    # Filter by year
    if year_start and 'year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['year'] >= year_start]
    if year_end and 'year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['year'] <= year_end]
    
    return filtered_df

def analyze_songs(df, group_name, output_dir):
    """
    Analyze a group of songs and save results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute repetition scores for each song
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Analyzing {group_name}"):
        try:
            if pd.isna(row['phonemes']) or not isinstance(row['phonemes'], str):
                continue
                
            # Calculate repeat scores
            scores = compute_repeat_scores(row['phonemes'])
            
            # Gather song metadata
            result = {
                'song_id': row.get('song_id', idx),
                'title': row.get('title', f"Song {idx}"),
                'artist': row.get('artist', 'Unknown'),
                'language': row.get('language', 'Unknown'),
                'year': row.get('year', None),
                'genre': row.get('tag', row.get('genre', 'Unknown')),
                'nw_score': scores['nw_score'],
                'dtw_score': scores['dtw_score'],
                'lcs_score': scores['lcs_score'],
                'combined_score': scores['combined_score'],
                'line_count': scores['line_count']
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing song {idx}: {e}")
    
    # Create results DataFrame
    if not results:
        print(f"No results for {group_name}")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Save results
    safe_name = re.sub(r'[^\w]', '_', group_name.lower())
    results_path = os.path.join(output_dir, f"{safe_name}_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Create summary statistics
    summary = {
        'Group': group_name,
        'Count': len(results_df),
        'Mean_NW': results_df['nw_score'].mean(),
        'Mean_DTW': results_df['dtw_score'].mean(),
        'Mean_LCS': results_df['lcs_score'].mean(),
        'Mean_Combined': results_df['combined_score'].mean(),
        'Std_Combined': results_df['combined_score'].std()
    }
    
    return results_df, summary

def create_comparison_plots(results_dict, output_dir):
    """
    Create comparison plots between different groups
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all summaries
    all_summaries = []
    for group_name, (_, summary) in results_dict.items():
        all_summaries.append(summary)
    
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)
    
    # Create bar plot for combined scores
    plt.figure(figsize=(12, 6))
    groups = summary_df['Group']
    scores = summary_df['Mean_Combined']
    errors = summary_df['Std_Combined']
    
    plt.bar(groups, scores, yerr=errors, capsize=5)
    plt.title('Comparison of Combined Repetition Scores')
    plt.ylabel('Combined Score (Normalized)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_scores_comparison.png"))
    plt.close()
    
    # Create comparison bar chart for individual algorithms
    plt.figure(figsize=(12, 8))
    bar_width = 0.25
    index = np.arange(len(groups))
    
    plt.bar(index - bar_width, summary_df['Mean_NW'], bar_width, label='Needleman-Wunsch')
    plt.bar(index, summary_df['Mean_DTW'], bar_width, label='Dynamic Time Warping')
    plt.bar(index + bar_width, summary_df['Mean_LCS'], bar_width, label='Longest Common Subsequence')
    
    plt.title('Comparison of Repetition Scores by Algorithm')
    plt.ylabel('Score (Normalized)')
    plt.xticks(index, groups, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"))
    plt.close()
    
    # Create heatmap for algorithm scores across groups
    score_data = summary_df[['Mean_NW', 'Mean_DTW', 'Mean_LCS']].copy()
    score_data.index = summary_df['Group']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(score_data, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Heatmap of Repetition Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_heatmap.png"))
    plt.close()

def compare_by_language(df, languages, genre=None, output_dir='../dataset/repeat_analysis'):
    """
    Compare repetition scores across different languages within the same genre
    """
    results = {}
    summaries = []
    
    for language in languages:
        filtered_df = filter_songs_by_criteria(df, language=language, genre=genre)
        group_name = f"{language}" + (f" - {genre}" if genre else "")
        
        if len(filtered_df) > 0:
            result = analyze_songs(filtered_df, group_name, output_dir)
            if result:
                results[group_name] = result
                summaries.append(result[1])
    
    # Create comparison visualizations
    create_comparison_plots(results, os.path.join(output_dir, 'language_comparison'))
    
    return results

def compare_by_genre(df, genres, language=None, output_dir='../dataset/repeat_analysis'):
    """
    Compare repetition scores across different genres within the same language
    """
    results = {}
    summaries = []
    
    for genre in genres:
        filtered_df = filter_songs_by_criteria(df, language=language, genre=genre)
        group_name = f"{genre}" + (f" - {language}" if language else "")
        
        if len(filtered_df) > 0:
            result = analyze_songs(filtered_df, group_name, output_dir)
            if result:
                results[group_name] = result
                summaries.append(result[1])
    
    # Create comparison visualizations
    create_comparison_plots(results, os.path.join(output_dir, 'genre_comparison'))
    
    return results

def compare_by_year(df, year_ranges, language=None, genre=None, output_dir='../dataset/repeat_analysis'):
    """
    Compare repetition scores across different time periods
    """
    results = {}
    summaries = []
    
    for start_year, end_year in year_ranges:
        filtered_df = filter_songs_by_criteria(
            df, language=language, genre=genre, 
            year_start=start_year, year_end=end_year
        )
        
        group_name = f"{start_year}-{end_year}"
        if language:
            group_name += f" - {language}"
        if genre:
            group_name += f" - {genre}"
        
        if len(filtered_df) > 0:
            result = analyze_songs(filtered_df, group_name, output_dir)
            if result:
                results[group_name] = result
                summaries.append(result[1])
    
    # Create comparison visualizations
    create_comparison_plots(results, os.path.join(output_dir, 'year_comparison'))
    
    return results

def main():
    # Configuration
    # Load the processed data
    # processed_dir = '../dataset/processed'
    # df = load_processed_data(processed_dir)
    
    # print(f"Loaded {len(df)} rows from processed data")

    df = pd.read_csv("../dataset/10k/score.csv")

    output_dir = '../dataset/repeat_analysis_10k'
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure necessary columns exist
    if 'phonemes' not in df.columns:
        print("Error: 'phonemes' column not found in dataset. Make sure the dataset has been processed.")
        return
    
    # Filter for rows with phonemes
    df = df[df['phonemes'].notna()]
    print(f"Found {len(df)} songs with phonemes")
    
    # Set up comparative analysis
    
    # 1. Compare different languages with same genre
    print("\n=== Comparing languages with same genre ===")
    languages_to_compare = ['en-us', 'fr-fr', 'de']  # Common languages
    genre_for_comparison = 'rap'  #  genre
    language_results = compare_by_language(
        df, languages_to_compare, genre=genre_for_comparison, 
        output_dir=output_dir
    )
    
    # 2. Compare different genres with same language
    print("\n=== Comparing genres with same language ===")
    genres_to_compare = ['rock', 'rap', 'pop']  # Common genres
    language_for_comparison = 'en-us'  #  language
    genre_results = compare_by_genre(
        df, genres_to_compare, language=language_for_comparison, 
        output_dir=output_dir
    )
    
    # 3. Compare songs from different time periods
    print("\n=== Comparing songs from different time periods ===")
    year_ranges = [
        (1950, 1969),
        (1970, 1989),
        (1990, 2009),
        (2010, 2023)
    ]
    time_results = compare_by_year(
        df, year_ranges, language=language_for_comparison,
        output_dir=output_dir
    )
    
    # Generate overall report
    summary_report = {
        "language_comparison": {
            "question": "Do phonetic features affect lyrics in different languages with the same genre?",
            "findings": "See detailed analysis in language_comparison directory",
            "groups_analyzed": len(language_results)
        },
        "genre_comparison": {
            "question": "Do phonetic features affect lyrics in different genres with the same language?",
            "findings": "See detailed analysis in genre_comparison directory",
            "groups_analyzed": len(genre_results)
        },
        "year_comparison": {
            "question": "Does the release year of a song affect its dependency on phonetic features?",
            "findings": "See detailed analysis in year_comparison directory",
            "groups_analyzed": len(time_results)
        }
    }
    
    # Save summary report
    with open(os.path.join(output_dir, "report_summary.txt"), "w") as f:
        f.write("=== Repeat Analysis Summary ===\n\n")
        
        for section, details in summary_report.items():
            f.write(f"Question: {details['question']}\n")
            f.write(f"Groups analyzed: {details['groups_analyzed']}\n")
            f.write(f"Findings: {details['findings']}\n\n")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
