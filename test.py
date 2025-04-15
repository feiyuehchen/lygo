from transphone import read_tokenizer                                                                                                  
import pandas as pd
from langdetect import detect
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

tokenizers = {
    'en': read_tokenizer('en'),
}


def detect_language(text: str) -> dict:
    return detect(text)


def text2phoneme(lyrics):
    lang = detect_language(lyrics)
    # use 2-char or 3-char ISO id to specify your target language 
    if lang not in tokenizers:
        try:
            tokenizer = read_tokenizer(lang)    
            tokenizers[lang] = tokenizer
        except Exception as e:
            print(f"language {lang} not supported")
            return None, None
    else:
        tokenizer = tokenizers[lang]

    phonemes = tokenizer.tokenize(lyrics) 
    
    return phonemes, lang


def process_row(row_data):
    """Process a single row of data."""
    lyrics = row_data['lyrics']
    phonemes, lang = text2phoneme(lyrics)
    row_data['phonemes'] = phonemes
    row_data['language'] = lang
    return row_data


def process_dataset_in_chunks(path: str, output_dir: str, chunk_size=100000, num_processes=None):
    """
    Read a CSV file in chunks, process each chunk using multiprocessing, and save to separate files.
    
    Args:
        path: Path to the CSV file
        output_dir: Directory to save the processed chunks
        chunk_size: Number of rows per chunk
        num_processes: Number of processes to use (default: number of CPU cores)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Read CSV in chunks
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        print(f"Processing chunk {i} with {num_processes} processes")
        
        # Convert chunk to list of dictionaries for multiprocessing
        rows_to_process = chunk.to_dict('records')
        
        # Process using multiprocessing pool with progress bar
        with mp.Pool(processes=num_processes) as pool:
            processed_rows = list(tqdm(
                pool.imap(process_row, rows_to_process),
                total=len(rows_to_process),
                desc=f"Chunk {i}"
            ))
        
        # Create a new DataFrame with processed data
        processed_chunk = pd.DataFrame(processed_rows)
        
        # Save to file
        output_path = os.path.join(output_dir, f"processed_chunk_{i}.csv")
        processed_chunk.to_csv(output_path, index=False)
        
        print(f"Saved chunk {i} to {output_path}")


if __name__ == '__main__':
    # text = 'hello world\n hey hey \n [Chorus]'
    # print(text2phoneme(text))
    dataset_path = '../dataset/song_lyrics.csv'
    output_dir = '../dataset/processed'
    
    # Use multiprocessing with a specified number of processes or default to CPU count
    process_dataset_in_chunks(dataset_path, output_dir, num_processes=32)  # Adjust number as needed