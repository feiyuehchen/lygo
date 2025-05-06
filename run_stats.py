import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from math import log

def load_processed_data(processed_dir):
    """Load all processed chunks and combine them into a single DataFrame"""
    print("Loading processed data...")
    chunk_files = glob.glob(os.path.join(processed_dir, "processed_chunk_*.csv"))
    
    # Sort files by chunk number
    chunk_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    dfs = []
    for file in chunk_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
        
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("No data could be loaded from the processed chunks")

def analyze_transformation_success(df):
    """Analyze the success rate of phoneme transformation"""
    total_rows = len(df)
    
    # Count successful transformations (non-null phonemes)
    successful = df['phonemes'].notna().sum()
    success_rate = successful / total_rows * 100
    
    # Create a dictionary of metrics
    metrics = {
        'Total Songs': total_rows,
        'Successful Transformations': successful,
        'Failed Transformations': total_rows - successful,
        'Success Rate (%)': success_rate
    }
    
    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie([metrics['Successful Transformations'], metrics['Failed Transformations']], 
            labels=['Success', 'Failed'],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            explode=(0.05, 0))
    plt.title('Phoneme Transformation Success Rate')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/transformation_success.png')
    plt.close()
    
    # Create a bar chart showing successful and failed transformations
    plt.figure(figsize=(8, 6))
    labels = ['Successful', 'Failed']
    values = [metrics['Successful Transformations'], metrics['Failed Transformations']]
    plt.bar(labels, values, color=['#4CAF50', '#F44336'])
    plt.title('Phoneme Transformation Count')
    plt.ylabel('Number of Songs')
    plt.yscale('log')  # Use logarithmic scale
    for i, v in enumerate(values):
        plt.text(i, v * 1.1, f"{v}", ha='center')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/transformation_count.png')
    plt.close()
    
    return metrics

def analyze_length_stats(df):
    """Calculate average length statistics for lyrics and phonemes"""
    # Filter for rows with successful transformations
    successful_df = df[df['phonemes'].notna()]
    
    # Calculate lengths
    successful_df['lyrics_length'] = successful_df['lyrics'].str.len()
    successful_df['phonemes_length'] = successful_df['phonemes'].str.len()
    
    # Calculate average lengths
    avg_lyrics_length = successful_df['lyrics_length'].mean()
    avg_phonemes_length = successful_df['phonemes_length'].mean()
    
    # Calculate ratio of phonemes to lyrics length
    successful_df['length_ratio'] = successful_df['phonemes_length'] / successful_df['lyrics_length']
    avg_ratio = successful_df['length_ratio'].mean()
    
    # Create metrics dictionary
    metrics = {
        'Average Lyrics Length': avg_lyrics_length,
        'Average Phonemes Length': avg_phonemes_length,
        'Average Phonemes/Lyrics Ratio': avg_ratio
    }
    
    # Create histograms for length distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Lyrics length histogram
    axes[0].hist(successful_df['lyrics_length'].clip(0, 5000), bins=50, color='#2196F3', alpha=0.7)
    axes[0].axvline(avg_lyrics_length, color='r', linestyle='dashed', linewidth=1)
    axes[0].set_title('Lyrics Length Distribution')
    axes[0].set_xlabel('Length (characters)')
    axes[0].set_ylabel('Count')
    axes[0].text(0.95, 0.95, f'Avg: {avg_lyrics_length:.1f}', 
                 transform=axes[0].transAxes, 
                 verticalalignment='top', 
                 horizontalalignment='right')
    
    # Phonemes length histogram
    axes[1].hist(successful_df['phonemes_length'].clip(0, 5000), bins=50, color='#FF9800', alpha=0.7)
    axes[1].axvline(avg_phonemes_length, color='r', linestyle='dashed', linewidth=1)
    axes[1].set_title('Phonemes Length Distribution')
    axes[1].set_xlabel('Length (characters)')
    axes[1].set_ylabel('Count')
    axes[1].text(0.95, 0.95, f'Avg: {avg_phonemes_length:.1f}', 
                 transform=axes[1].transAxes, 
                 verticalalignment='top', 
                 horizontalalignment='right')
    
    # Ratio histogram
    axes[2].hist(successful_df['length_ratio'].clip(0, 5), bins=50, color='#4CAF50', alpha=0.7)
    axes[2].axvline(avg_ratio, color='r', linestyle='dashed', linewidth=1)
    axes[2].set_title('Phonemes/Lyrics Length Ratio')
    axes[2].set_xlabel('Ratio')
    axes[2].set_ylabel('Count')
    axes[2].text(0.95, 0.95, f'Avg: {avg_ratio:.2f}', 
                 transform=axes[2].transAxes, 
                 verticalalignment='top', 
                 horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/length_stats.png')
    plt.close()
    
    # Create a scatter plot of lyrics length vs phonemes length
    plt.figure(figsize=(10, 6))
    plt.scatter(
        successful_df['lyrics_length'].clip(0, 2000), 
        successful_df['phonemes_length'].clip(0, 2000), 
        alpha=0.1, color='#2196F3'
    )
    plt.xlabel('Lyrics Length (characters)')
    plt.ylabel('Phonemes Length (characters)')
    plt.title('Lyrics Length vs Phonemes Length')
    
    # Add a regression line
    x = successful_df['lyrics_length'].clip(0, 2000)
    y = successful_df['phonemes_length'].clip(0, 2000)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--', alpha=0.8)
    plt.text(0.95, 0.05, f'y = {z[0]:.2f}x + {z[1]:.2f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='bottom', 
             horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/length_correlation.png')
    plt.close()
    
    return metrics

def analyze_language_stats(df):
    """Analyze language statistics"""
    # Filter for rows with successful transformations
    successful_df = df[df['phonemes'].notna()]
    
    # Count languages
    language_counts = successful_df['language'].value_counts()
    
    # Create a bar chart for all languages with count as y-axis on logarithmic scale
    plt.figure(figsize=(14, 8))
    language_counts.plot(kind='bar', color=sns.color_palette('viridis', len(language_counts)))
    plt.title('Language Distribution in Dataset (Log Scale)')
    plt.xlabel('Language')
    plt.ylabel('Count (Log Scale)')
    plt.yscale('log')  # Use logarithmic scale for y-axis
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/language_distribution_log.png')
    plt.close()
    
    # Also create a regular scale version
    plt.figure(figsize=(14, 8))
    language_counts.plot(kind='bar', color=sns.color_palette('viridis', len(language_counts)))
    plt.title('Language Distribution in Dataset')
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/language_distribution_all.png')
    plt.close()
    
    # Calculate phoneme/word ratio for each language
    language_phoneme_word_ratios = {}
    for lang in language_counts.index:
        lang_df = successful_df[successful_df['language'] == lang]
        
        # Calculate word count for each song
        lang_df['word_count'] = lang_df['lyrics'].apply(lambda x: len(str(x).split()))
        
        # Calculate phoneme/word ratio
        lang_df['phoneme_word_ratio'] = lang_df['phonemes'].str.len() / lang_df['word_count']
        
        # Store average ratio
        avg_ratio = lang_df['phoneme_word_ratio'].mean()
        language_phoneme_word_ratios[lang] = avg_ratio
    
    # Create bar chart for phoneme/word ratio by language
    plt.figure(figsize=(14, 8))
    languages = list(language_phoneme_word_ratios.keys())
    ratios = list(language_phoneme_word_ratios.values())
    
    # Sort by ratio
    sorted_indices = np.argsort(ratios)[::-1]
    sorted_languages = [languages[i] for i in sorted_indices]
    sorted_ratios = [ratios[i] for i in sorted_indices]
    
    plt.bar(sorted_languages, sorted_ratios, color=sns.color_palette('coolwarm', len(sorted_languages)))
    plt.title('Average Phoneme/Word Ratio by Language')
    plt.xlabel('Language')
    plt.ylabel('Phoneme/Word Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/phoneme_word_ratio_by_language.png')
    plt.close()
    
    # Create a version with log scale for the ratio
    plt.figure(figsize=(14, 8))
    plt.bar(sorted_languages, sorted_ratios, color=sns.color_palette('coolwarm', len(sorted_languages)))
    plt.title('Average Phoneme/Word Ratio by Language (Log Scale)')
    plt.xlabel('Language')
    plt.ylabel('Phoneme/Word Ratio (Log Scale)')
    plt.yscale('log')  # Use logarithmic scale for y-axis
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/phoneme_word_ratio_by_language_log.png')
    plt.close()
    
    return language_counts, language_phoneme_word_ratios

def analyze_genre_stats(df):
    """Analyze genre statistics"""
    # Filter for rows with successful transformations and genre information
    if 'tag' not in df.columns:
        print("No genre column found in dataset. Skipping genre analysis.")
        return None
    
    successful_df = df[df['phonemes'].notna()]
    
    # Count genres
    genre_counts = successful_df['tag'].value_counts()
    # print(genre_counts)
    
    # Create a bar chart for genres with count as y-axis on logarithmic scale
    plt.figure(figsize=(14, 8))
    genre_counts.plot(kind='bar', color=sns.color_palette('viridis', len(genre_counts)))
    plt.title('Genre Distribution in Dataset (Log Scale)')
    plt.xlabel('Genre')
    plt.ylabel('Count (Log Scale)')
    plt.yscale('log')  # Use logarithmic scale for y-axis
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/genre_distribution_log.png')
    plt.close()
    
    # Also create a regular scale version
    plt.figure(figsize=(14, 8))
    genre_counts.plot(kind='bar', color=sns.color_palette('viridis', len(genre_counts)))
    plt.title('Genre Distribution in Dataset')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../dataset/stats_10000/genre_distribution.png')
    plt.close()
    
    return genre_counts

def main():
    # Create stats directory if it doesn't exist
    os.makedirs('../dataset/stats_10000', exist_ok=True)
    
    # Load the processed data
    # processed_dir = '../dataset/10k'
    # data = load_processed_data(processed_dir)
    data = pd.read_csv('../dataset/10k/score.csv')
    
    print(f"Loaded {len(data)} rows from processed data")
    
    # Check if phonemes column exists
    if 'phonemes' not in data.columns:
        raise ValueError("The dataset doesn't have the expected 'phonemes' column")
    
    # Run the analyses
    print("Analyzing transformation success...")
    transform_metrics = analyze_transformation_success(data)
    print(f"Successful transformations: {transform_metrics['Successful Transformations']} out of {transform_metrics['Total Songs']}")
    print(f"Success rate: {transform_metrics['Success Rate (%)']:.2f}%")
    
    print("Analyzing language statistics...")
    language_counts, language_phoneme_word_ratios = analyze_language_stats(data)
    print(f"Found {len(language_counts)} different languages in the dataset")
    
    print("Analyzing genre statistics...")
    genre_counts = analyze_genre_stats(data)
    if genre_counts is not None:
        print(f"Found {len(genre_counts)} different genres in the dataset")
    
    # Save summary metrics
    print("Saving summary metrics...")
    
    # Prepare summary data
    summary = {
        'Metric': [
            'Total Songs',
            'Successful Transformations',
            'Failed Transformations',
            'Success Rate (%)',
            'Number of Languages',
        ],
        'Value': [
            transform_metrics['Total Songs'],
            transform_metrics['Successful Transformations'],
            transform_metrics['Failed Transformations'],
            transform_metrics['Success Rate (%)'],
            len(language_counts),
        ]
    }
    
    # Add genre count if available
    if genre_counts is not None:
        summary['Metric'].append('Number of Genres')
        summary['Value'].append(len(genre_counts))
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('../dataset/stats_10000/summary_metrics.csv', index=False)
    
    print("Analysis complete! Statistics saved to ../dataset/stats_10000/")

if __name__ == "__main__":
    main()