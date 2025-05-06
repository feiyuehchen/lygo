import os
from utils import *


if __name__ == "__main__":
    processed_dir = '../dataset/processed'
    output_path = "../dataset/genius_420k.xlsx"
    
    # os.makedirs(output_dir, exist_ok=True)
    # print("Loading processed data...")
    # chunk_files = glob.glob(os.path.join(processed_dir, "*.csv"))

    # for chunk_file in chunk_files:
    #     df = pd.read_csv(chunk_file)

    df = load_processed_data(processed_dir)

    # freq = df['language'].value_counts()
    # sorted_df = df.sort_values(by='language', key=lambda x: x.map(freq), ascending=False)

    # top_5_languages = df['language'].value_counts().nlargest(5).index.tolist()

    # top_5_genres = df['tag'].value_counts().nlargest(5).index.tolist()

    # filtered_df = df[(df['language'].isin(top_5_languages)) & (df['tag'].isin(top_5_genres))]

    # if len(filtered_df) > 10000:
    #     result_df = filtered_df.sample(n=10000, random_state=42)
    # else:
    #     result_df = filtered_df
    #     print(f"Warning: Only {len(filtered_df)} rows available with the top 5 languages and genres.")

    df = df.drop(columns=['lyrics'])
    print(df.columns)
    df.to_excel(output_path, index=False)

    

    



    