import glob
import os
import pandas as pd
import numpy as np  
from tqdm import tqdm

def load_processed_data(processed_dir):
    """Load all processed chunks and combine them into a single DataFrame"""
    print("Loading processed data...")
    chunk_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    # Sort files by chunk number
    chunk_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    dfs = []
    for file in tqdm(chunk_files):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            # print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

        # break
        
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("No data could be loaded from the processed chunks")




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

def compute_repeat_scores(phonemes):
    """
    Compute repetition scores between consecutive sentences in phoneme representation
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
    
    # Calculate average scores for the song
    avg_nw = np.mean(nw_scores) if nw_scores else 0
    avg_dtw = np.mean(dtw_scores) if dtw_scores else 0
    avg_lcs = np.mean(lcs_scores) if lcs_scores else 0
    
    # Calculate combined score (average of all three algorithms)
    combined_score = (avg_nw + avg_dtw + avg_lcs) / 3
    
    return {
        'nw_score': avg_nw,
        'dtw_score': avg_dtw,
        'lcs_score': avg_lcs,
        'combined_score': combined_score,
        'line_count': len(phoneme_lines)
    }


def compute_dtw_scores(phonemes):
    """
    Compute repetition scores between consecutive sentences in phoneme representation
    """
    # Split phonemes into lines
    phoneme_lines = phonemes.strip().split('\n')
    
    # Filter out empty lines
    phoneme_lines = [line for line in phoneme_lines if line.strip()]
    
    # Compute similarity scores between consecutive lines
    # nw_scores = []
    dtw_scores = []
    # lcs_scores = []
    
    for i in range(len(phoneme_lines) - 1):
        seq1 = phoneme_lines[i]
        seq2 = phoneme_lines[i + 1]
        
        # Calculate scores
        # nw_score = needleman_wunsch(seq1, seq2)
        dtw_score = dynamic_time_warping(seq1, seq2)
        # lcs_score = longest_common_subsequence(seq1, seq2)
        
        # nw_scores.append(nw_score)
        dtw_scores.append(dtw_score)
        # lcs_scores.append(lcs_score)
    
    # Calculate average scores for the song
    # avg_nw = np.mean(nw_scores) if nw_scores else 0
    avg_dtw = np.mean(dtw_scores) if dtw_scores else 0
    # avg_lcs = np.mean(lcs_scores) if lcs_scores else 0
    
    # Calculate combined score (average of all three algorithms)
    # combined_score = (avg_nw + avg_dtw + avg_lcs) / 3
    
    return {
        # 'nw_score': avg_nw,
        'dtw_score': avg_dtw,
        # 'lcs_score': avg_lcs,
        # 'combined_score': combined_score,
        # 'line_count': len(phoneme_lines)
    }