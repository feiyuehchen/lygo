import pandas as pd
import time
from openai import OpenAI
from tqdm import tqdm

# Initialize the OpenAI client
api_key = "OPENAI_API_KEY"
client = OpenAI(api_key=api_key)

# Read the dataset (adjust the path as needed)
def read_chunk(file_path):
    return pd.read_csv(file_path)

def generate_lyrics(artist, genre):
    prompt = f"""
    Write original song lyrics in the style of {artist}, within the {genre} genre. 
    Do not use any additional themes, topics, or context—simply draw from the artist's typical lyricism, word choice, and the conventions of the specified genre. 
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=2048,
            top_p=1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating lyrics for {artist} in {genre}: {e}")
        return None

def main():
    # Path to your data file
    data_path = "/home/b06611012/temp/dataset/processed/processed_chunk_0.csv"
    
    # Read the dataset
    df = read_chunk(data_path)
    
    # Extract unique artist-genre pairs
    artist_genre_pairs = df[['artist', 'tag']].drop_duplicates()
    
    # Take the first 100 pairs
    pairs_to_process = artist_genre_pairs.head(100)
    
    # Create a new dataframe to store results
    results = []
    
    # Generate lyrics for each pair
    for _, row in tqdm(pairs_to_process.iterrows(), total=len(pairs_to_process)):
        artist = row['artist']
        genre = row['tag']
        
        lyrics = generate_lyrics(artist, genre)
        
        if lyrics:
            results.append({
                'artist': artist,
                'tag': genre,
                'lyrics': lyricsP
            })
        
        # Add a small delay to avoid rate limiting
        # time.sleep(1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv("generated_lyrics.csv", index=False)
    print(f"Generated lyrics for {len(results)} artist-genre pairs")

if __name__ == "__main__":
    main()