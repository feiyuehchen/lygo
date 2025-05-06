import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-1):
    """
    Implement Needleman-Wunsch algorithm for sequence alignment
    Returns the alignment score between two sequences
    """
    # Initialize the scoring matrix
    n, m = len(seq1), len(seq2)
    score_matrix = np.zeros((n+1, m+1))
    
    # Initialize first row and column with gap penalties
    for i in range(n+1):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(m+1):
        score_matrix[0, j] = j * gap_penalty
    
    # Fill the scoring matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = score_matrix[i-1, j] + gap_penalty
            insert = score_matrix[i, j-1] + gap_penalty
            score_matrix[i, j] = max(match, delete, insert)
    
    # Normalize the score by dividing by the length of the longer sequence
    max_len = max(n, m)
    normalized_score = score_matrix[n, m] / max_len if max_len > 0 else 0
    
    return normalized_score

def dynamic_time_warping(seq1, seq2):
    """
    Implement Dynamic Time Warping for sequence alignment
    Returns the minimum distance between two sequences
    """
    n, m = len(seq1), len(seq2)
    
    # Create cost matrix
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = float('inf')
    dtw_matrix[0, 0] = 0
    
    # Fill the matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match or substitution
            )
    
    # Normalize the score
    max_len = max(n, m)
    normalized_score = 1 - (dtw_matrix[n, m] / max_len) if max_len > 0 else 0
    
    return normalized_score

def longest_common_subsequence(seq1, seq2):
    """
    Implement Longest Common Subsequence algorithm
    Returns the length of the LCS between two sequences
    """
    n, m = len(seq1), len(seq2)
    
    # Initialize the DP table
    lcs = [[0 for _ in range(m+1)] for _ in range(n+1)]
    
    # Fill the DP table
    for i in range(1, n+1):
        for j in range(1, m+1):
            if seq1[i-1] == seq2[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
    
    # Normalize the score
    max_len = max(n, m)
    normalized_score = lcs[n][m] / max_len if max_len > 0 else 0
    
    return normalized_score

def analyze_phoneme_repetitions(phonemes, language=None):
    """
    Analyze phonetic repetitions between consecutive lines of phonemes
    """
    # Split phonemes into lines
    phoneme_lines = phonemes.strip().split('\n')
    
    # Filter out empty lines
    phoneme_lines = [line for line in phoneme_lines if line.strip()]
    
    # Compute similarity scores between consecutive lines
    nw_scores = []
    dtw_scores = []
    lcs_scores = []
    
    for i in range(len(phoneme_lines) - 1):
        seq1 = phoneme_lines[i]
        seq2 = phoneme_lines[i + 1]
        
        # Calculate scores
        nw_score = needleman_wunsch(seq1, seq2)
        dtw_score = dynamic_time_warping(seq1, seq2)
        lcs_score = longest_common_subsequence(seq1, seq2)
        
        nw_scores.append(nw_score)
        dtw_scores.append(dtw_score)
        lcs_scores.append(lcs_score)
    
    return {
        'num_lines': len(phoneme_lines),
        'nw_scores': nw_scores,
        'dtw_scores': dtw_scores,
        'lcs_scores': lcs_scores,
        'avg_nw_score': np.mean(nw_scores) if nw_scores else 0,
        'avg_dtw_score': np.mean(dtw_scores) if dtw_scores else 0,
        'avg_lcs_score': np.mean(lcs_scores) if lcs_scores else 0
    }

def analyze_lyrics_dataset(dataset_path, output_dir, sample_size=None):
    """
    Analyze repetitions in a dataset of lyrics using precomputed phonemes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Take a sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Add columns for results
    df['avg_nw_score'] = 0.0
    df['avg_dtw_score'] = 0.0
    df['avg_lcs_score'] = 0.0
    
    # Process each song
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing songs"):
        try:
            phonemes = row['phonemes']
            language = row.get('language', None)
            
            if not pd.isna(phonemes) and isinstance(phonemes, str):
                # Analyze the phonemes
                analysis = analyze_phoneme_repetitions(phonemes, language)
                
                # Store aggregate scores
                df.loc[idx, 'avg_nw_score'] = analysis['avg_nw_score']
                df.loc[idx, 'avg_dtw_score'] = analysis['avg_dtw_score']
                df.loc[idx, 'avg_lcs_score'] = analysis['avg_lcs_score']
                
                # Collect results for further analysis
                result = {
                    'song_id': row.get('song_id', idx),
                    'title': row.get('title', f"Song {idx}"),
                    'artist': row.get('artist', 'Unknown'),
                    'language': language,
                    'genre': row.get('genre', 'Unknown'),
                    'year': row.get('year', None),
                    'avg_nw_score': analysis['avg_nw_score'],
                    'avg_dtw_score': analysis['avg_dtw_score'],
                    'avg_lcs_score': analysis['avg_lcs_score'],
                    'num_lines': analysis['num_lines']
                }
                results.append(result)
        except Exception as e:
            print(f"Error processing song {idx}: {e}")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    # Save the results
    df.to_csv(os.path.join(output_dir, "lyrics_with_scores.csv"), index=False)
    results_df.to_csv(os.path.join(output_dir, "repetition_analysis.csv"), index=False)
    
    # Generate plots by language
    if 'language' in results_df.columns:
        language_scores = results_df.groupby('language')[['avg_nw_score', 'avg_dtw_score', 'avg_lcs_score']].mean()
        
        plt.figure(figsize=(12, 6))
        language_scores.plot(kind='bar')
        plt.title('Average Repetition Scores by Language')
        plt.xlabel('Language')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "repetition_by_language.png"))
    
    # Generate plots by genre if available
    if 'genre' in results_df.columns and not results_df['genre'].isna().all():
        genre_scores = results_df.groupby('genre')[['avg_nw_score', 'avg_dtw_score', 'avg_lcs_score']].mean()
        
        plt.figure(figsize=(12, 6))
        genre_scores.plot(kind='bar')
        plt.title('Average Repetition Scores by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "repetition_by_genre.png"))
    
    # Generate plots by year if available
    if 'year' in results_df.columns and not results_df['year'].isna().all():
        year_scores = results_df.groupby('year')[['avg_nw_score', 'avg_dtw_score', 'avg_lcs_score']].mean()
        
        plt.figure(figsize=(12, 6))
        year_scores.plot(kind='line')
        plt.title('Average Repetition Scores by Year')
        plt.xlabel('Year')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "repetition_by_year.png"))
    
    return results_df

if __name__ == "__main__":
    # Configuration
    dataset_path = '../dataset/song_lyrics.csv'  # Path to the dataset with phonemes
    output_dir = '../dataset/analysis'
    sample_size = 1000  # Set to None to process the entire dataset
    
    # Run the analysis
    results = analyze_lyrics_dataset(dataset_path, output_dir, sample_size)
    
    print(f"Analysis complete! Results saved to {output_dir}")