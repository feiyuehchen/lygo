import pandas as pd
from langdetect import detect, DetectorFactory
import langid
import os
from tqdm import tqdm
import logging
from phonemizer import phonemize
import time
# from functools import lru_cache
import csv
from phonemizer.backend import EspeakBackend
import gc


# Set fixed seed to ensure consistent language detection results
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get all supported languages by espeak backend
# espeak_backend = EspeakBackend('espeak')
# SUPPORTED_ESPEAK_LANGUAGES = espeak_backend.supported_languages()




# Expanded language mapping with more languages
LANGUAGE_MAPPING = {
    # Core languages from original mapping
    'en': 'en-us',  # English (US)
    'zh': 'cmn',    # Chinese (Mandarin)
    'ja': 'ja',     # Japanese
    'ko': 'ko',     # Korean
    'fr': 'fr-fr',  # French
    'de': 'de',     # German
    'es': 'es',     # Spanish
    'it': 'it',     # Italian
    'ru': 'ru',     # Russian
    'ar': 'ar',     # Arabic
    'hi': 'hi',     # Hindi
    'pt': 'pt',     # Portuguese
    'nl': 'nl',     # Dutch
    'tr': 'tr',     # Turkish
    'pl': 'pl',     # Polish
    'sv': 'sv',     # Swedish
    'id': 'id',     # Indonesian
    'vi': 'vi',     # Vietnamese
    'th': 'th',     # Thai
    
    # Additional languages
    'af': 'af',     # Afrikaans
    'sq': 'sq',     # Albanian
    'am': 'am',     # Amharic
    'az': 'az',     # Azerbaijani
    'eu': 'eu',     # Basque
    'bn': 'bn',     # Bengali
    'bs': 'bs',     # Bosnian
    'bg': 'bg',     # Bulgarian
    'ca': 'ca',     # Catalan
    'hr': 'hr',     # Croatian
    'cs': 'cs',     # Czech
    'da': 'da',     # Danish
    'et': 'et',     # Estonian
    'fi': 'fi',     # Finnish
    'gl': 'gl',     # Galician
    'ka': 'ka',     # Georgian
    'el': 'el',     # Greek
    'gu': 'gu',     # Gujarati
    'he': 'he',     # Hebrew
    'hu': 'hu',     # Hungarian
    'is': 'is',     # Icelandic
    'kn': 'kn',     # Kannada
    'kk': 'kk',     # Kazakh
    'km': 'km',     # Khmer
    'ky': 'ky',     # Kyrgyz
    'lo': 'lo',     # Lao
    'lv': 'lv',     # Latvian
    'lt': 'lt',     # Lithuanian
    'mk': 'mk',     # Macedonian
    'ms': 'ms',     # Malay
    'ml': 'ml',     # Malayalam
    'mr': 'mr',     # Marathi
    'mn': 'mn',     # Mongolian
    'my': 'my',     # Myanmar (Burmese)
    'ne': 'ne',     # Nepali
    'no': 'no',     # Norwegian
    'fa': 'fa',     # Persian
    'pt-br': 'pt-br', # Portuguese (Brazil)
    'pa': 'pa',     # Punjabi
    'ro': 'ro',     # Romanian
    'si': 'si',     # Sinhala
    'sk': 'sk',     # Slovak
    'sl': 'sl',     # Slovenian
    'so': 'so',     # Somali
    'es-la': 'es-la', # Spanish (Latin America)
    'sw': 'sw',     # Swahili
    'ta': 'ta',     # Tamil
    'te': 'te',     # Telugu
    'tl': 'tl',     # Filipino
    'uk': 'uk',     # Ukrainian
    'ur': 'ur',     # Urdu
    'uz': 'uz',     # Uzbek
    'cy': 'cy',     # Welsh
    'zu': 'zu',     # Zulu
}

# # Filter the mapping to only include languages supported by espeak
# LANGUAGE_MAPPING = {k: v for k, v in LANGUAGE_MAPPING.items() 
#                    if v in SUPPORTED_ESPEAK_LANGUAGES or 
#                    any(v.startswith(lang) for lang in SUPPORTED_ESPEAK_LANGUAGES)}

logger.info(f"Total supported languages: {len(LANGUAGE_MAPPING)}")
logger.info(f"Supported languages: {', '.join(sorted(LANGUAGE_MAPPING.keys()))}")

# @lru_cache(maxsize=10000)
def detect_language(text: str) -> str:
    """Detect language using multiple methods and return mapped language code"""
    try:
        if not text or len(text.strip()) < 3:
            return 'en-us'  # Default to English
        
        # Use two different language detection libraries
        # lang_id = langid.classify(text)[0]
        lang_detect = detect(text)
        
        # Prioritize langid results as it performs better for short texts
        # detected_lang = lang_id
        detected_lang = lang_detect
        
        # Map to phonemizer supported language code
        if detected_lang in LANGUAGE_MAPPING:
            return LANGUAGE_MAPPING[detected_lang]
        else:
            logger.warning(f"Unsupported language: {detected_lang}, defaulting to English")
            return 'en-us'
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}, defaulting to English")
        return 'en-us'

# @lru_cache(maxsize=5000)
def text2phoneme(lyrics):
    """Convert text to phoneme with better error handling"""
    if not lyrics or pd.isna(lyrics):
        return None, None
        
    try:
        lang_code = detect_language(lyrics)
        
        phoneme_sentence = phonemize(
            lyrics,
            language=lang_code,
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
            preserve_empty_lines=True,
            words_mismatch='ignore',
            njobs=1,  # Reduce from 10 to 1
        )
        
    except Exception as e:
        logger.error(f"Phoneme conversion failed: {str(e)}")
        phoneme_sentence = None
        lang_code = None
    
    return phoneme_sentence, lang_code

def process_dataset_in_chunks(path: str, output_dir: str, chunk_size=1000, start_chunk=0):
    """Process CSV file in chunks with better error handling"""
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_index = 0
    processed_chunks = 0
    
    # Get total row count for progress tracking
    try:
        total_rows = sum(1 for _ in open(path, 'r', encoding='utf-8')) - 1
        logger.info(f"Total rows to process: {total_rows}")
    except Exception as e:
        logger.error(f"Error counting rows: {str(e)}")
        total_rows = 0
    
    # Process file in chunks
    backend = None
    files_processed = 0
    
    try:
        for chunk_df in pd.read_csv(path, chunksize=chunk_size, on_bad_lines='skip'):
            # Skip already processed chunks
            if chunk_index < start_chunk:
                chunk_index += 1
                continue
            
            logger.info(f"Processing chunk {chunk_index}")
            
            # Restart espeak backend every 1500 files
            if files_processed > 1000:
                logger.info("Restarting espeak backend to prevent resource issues")
                if backend:
                    del backend
                backend = EspeakBackend()
                files_processed = 0
                # Force garbage collection
                gc.collect()
            
            # Add phonemes and language columns if they don't exist
            if 'phonemes' not in chunk_df.columns:
                chunk_df['phonemes'] = None
            if 'language' not in chunk_df.columns:
                chunk_df['language'] = None
            
            # Process each row
            success_count = 0
            for i, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Chunk {chunk_index}"):
                try:
                    if 'lyrics' in chunk_df.columns and not pd.isna(row['lyrics']):
                        phonemes, lang = text2phoneme(row['lyrics'])
                        chunk_df.at[i, 'phonemes'] = phonemes
                        chunk_df.at[i, 'language'] = lang
                        success_count += 1
                except Exception as e:
                    # Just log and continue to next row
                    logger.error(f"Error in row {i}: {str(e)}")
            
            # Save chunk regardless of errors
            try:
                output_path = os.path.join(output_dir, f"processed_chunk_{chunk_index}.csv")
                chunk_df.to_csv(output_path, index=False)
                logger.info(f"Saved chunk {chunk_index} with {success_count}/{len(chunk_df)} successes")
            except Exception as e:
                logger.error(f"Error saving chunk {chunk_index}: {str(e)}")
            
            processed_chunks += 1
            chunk_index += 1
            files_processed += len(chunk_df)
            
            # Enable garbage collection after each chunk
            gc.collect()
            
            
    except Exception as e:
        logger.error(f"Error processing chunks: {str(e)}")
    
    
    logger.info(f"Processed {processed_chunks} chunks successfully")

def process_and_save_chunk(chunk_rows, header, chunk_index, output_dir, overall_pbar):
    """Process a chunk of rows and save to a CSV file"""
    logger.info(f"Processing chunk {chunk_index}, containing {len(chunk_rows)} rows")
    chunk_start_time = time.time()
    
    # Find 'lyrics' column index
    try:
        lyrics_index = header.index('lyrics')
    except ValueError:
        logger.error("Could not find 'lyrics' column in header")
        return
    
    # Process rows
    processed_rows = []
    for row in tqdm(chunk_rows, desc=f"Processing chunk {chunk_index}"):
        try:
            if lyrics_index < len(row):
                lyrics = row[lyrics_index]
                phonemes, lang = text2phoneme(lyrics)
                
                # Create new row with all fields plus phonemes and language
                new_row = row.copy()
                while len(new_row) < len(header) - 2:  # Add padding if row is short
                    new_row.append('')
                
                new_row.append(phonemes)
                new_row.append(lang)
                
                processed_rows.append(new_row)
            else:
                # Row doesn't have enough columns, add empty values for phonemes and language
                new_row = row.copy()
                while len(new_row) < len(header):
                    new_row.append('')
                processed_rows.append(new_row)
        except Exception as e:
            logger.error(f"Error processing row: {str(e)}")
            # Add row with empty phonemes and language
            new_row = row.copy()
            while len(new_row) < len(header) - 2:
                new_row.append('')
            new_row.append(None)
            new_row.append(None)
            processed_rows.append(new_row)
    
    # Save to file using CSV writer
    output_path = os.path.join(output_dir, f"processed_chunk_{chunk_index}.csv")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(processed_rows)
    
    # Update progress
    overall_pbar.update(len(chunk_rows))
    
    # Log completion
    chunk_duration = time.time() - chunk_start_time

    logger.info(f"Chunk {chunk_index} completed, took {chunk_duration:.2f} seconds, progress: {overall_pbar.n}/{overall_pbar.total}")


def combine_processed_chunks(output_dir: str, final_output_path: str):
    """Combine all processed chunks into a single file using memory-efficient approach"""
    logger.info(f"Starting to combine processed data chunks...")
    
    # Get all processed files
    chunk_files = sorted([f for f in os.listdir(output_dir) if f.startswith("processed_chunk_")])
    
    if not chunk_files:
        logger.error(f"No processed data chunks found in {output_dir}")
        return
    
    # Get header from first file
    with open(os.path.join(output_dir, chunk_files[0]), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    # Write combined file
    with open(final_output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        
        # Process each chunk file
        for chunk_file in tqdm(chunk_files, desc="Combining data chunks"):

            with open(os.path.join(output_dir, chunk_file), 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                
                # Write rows in batches to reduce memory usage
                batch_size = 1000
                batch = []
                
                for row in reader:
                    batch.append(row)
                    if len(batch) >= batch_size:
                        writer.writerows(batch)
                        batch = []
                
                # Write any remaining rows
                if batch:
                    writer.writerows(batch)

    
    logger.info(f"All data chunks combined into {final_output_path}")

if __name__ == '__main__':
    # Configuration
    dataset_path = '../dataset/song_lyrics.csv'
    output_dir = '../dataset/processed'
    
    # Resume from last completed chunk
    # Check which chunks already exist to determine start_chunk
    try:
        existing_chunks = [f for f in os.listdir(output_dir) if f.startswith("processed_chunk_")]
        if existing_chunks:
            chunk_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_chunks]
            start_chunk = max(chunk_numbers) + 1
            logger.info(f"Resuming from chunk {start_chunk}")
        else:
            start_chunk = 0
    except:
        start_chunk = 0
    
    # Process dataset
    process_dataset_in_chunks(
        path=dataset_path,
        output_dir=output_dir,
        chunk_size=1000,  # Smaller chunks to avoid memory issues
        start_chunk=start_chunk
    )
    
    # Optional: Combine all processed chunks
    # combine_processed_chunks(
    #     output_dir=output_dir,
    #     final_output_path='../dataset/lyrics_with_phonemes.csv'
    # )