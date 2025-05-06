from utils import *
# from glob import glob
from tqdm import tqdm
import pickle
import pandas as pd
import os

if __name__ == "__main__":
    # processed_dir = '../dataset/processed'
    # output_dir = "../dataset/output"
    # os.makedirs(output_dir, exist_ok=True)
    # print("Loading processed data...")
    # chunk_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    # Sort files by chunk number
    # chunk_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    file_path = "../dataset/chosen.csv"
    output_path = "../dataset/score.csv"
    df = pd.read_csv(file_path)
    df['dtw_score'] = None

    new_df = pd.DataFrame(columns=df.columns)

    print(f"process {file_path}")
    for id, row in tqdm(df.iterrows()):
        # scores = compute_repeat_scores(row["phonemes"])
        scores = compute_dtw_scores(row["phonemes"])
        # print(scores)
        # row['nw_score'] = scores['nw_score']
        row['dtw_score'] = scores['dtw_score']
        print(scores['dtw_score'])
        # row['lcs_score'] = scores['lcs_score']
        new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
        # break
    # output_path = os.path.join(output_dir, os.path.basename(chunk_file))
    new_df.to_csv(output_path, index=False)



