from __future__ import annotations
import re
import html
from typing import Dict, Optional, Tuple, Set, List
import pandas as pd

# Top English stopwords for fast, robust language detection
ENGLISH_STOPWORDS: Set[str] = {
    'the', 'and', 'is', 'to', 'in', 'my', 'it', 'for', 'you', 'on', 'with',
    'have', 'this', 'that', 'can', 'not', 'we', 'our', 'your', 'please', 'help',
    'at', 'from', 'be', 'are', 'was', 'so', 'me', 'what', 'all', 'would',
    'there', 'if', 'get', 'when', 'or', 'an', 'they', 'just', 'do', 'about'
}

def clean_text(text: str, is_brand_reply: bool = False) -> str:
    """
    Cleans tweet text for customer support NLP processing.
    - Decodes HTML entities (&amp; -> &)
    - Strips @mention tags
    - Normalizes multiple spaces and newlines
    - Removes brand agent signatures (e.g. ^AG, -Alex, /CH) from brand replies
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    text = text.replace('�', "'")
    
    # 2. If brand reply, strip trailing agent initials/signatures (e.g., ^TN, ^AG, -Sam, /LS)
    if is_brand_reply:
        text = re.sub(r'\s*([\^\/\-][A-Za-z]{1,4}|\*[A-Za-z]{1,4})\s*$', '', text)
    
    # 3. Strip @mentions (e.g., @115820, @AmazonHelp)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # 4. Normalize spaces and whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_predominantly_english(text: str, min_stopword_matches: int = 2) -> bool:
    """
    Checks if text is predominantly English without requiring heavy external dependencies.
    Evaluates ASCII character ratio and presence of common English stop words.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return False
    
    # Check ASCII character ratio
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / max(len(text), 1) < 0.80:
        return False
    
    # Tokenize words (lowercase alphabetic)
    words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    matches = words & ENGLISH_STOPWORDS
    
    return len(matches) >= min_stopword_matches

def extract_resolution_links(text: str) -> List[str]:
    """Extracts URLs from tweet text."""
    if not isinstance(text, str):
        return []
    return re.findall(r'https?://[\w\.\/\-]+', text)

def process_raw_dataset(
    csv_path: str,
    brand: str = 'AmazonHelp',
    max_pairs: int = 30000,
    chunksize: int = 100000
) -> pd.DataFrame:
    """
    Streams TWCS dataset, extracts conversation pairs for the target brand,
    filters to English, cleans text, and returns a structured DataFrame.
    """
    print(f"Loading and filtering {brand} tweets from {csv_path}...")
    
    # Step 1: Collect brand replies and parent tweet IDs
    brand_replies_list = []
    parent_ids_needed = set()
    
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'in_response_to_tweet_id'],
        low_memory=False
    ):
        b_mask = (chunk['inbound'] == False) & (chunk['author_id'] == brand) & chunk['in_response_to_tweet_id'].notna()
        b_replies = chunk[b_mask]
        
        if not b_replies.empty:
            brand_replies_list.append(b_replies)
            parent_ids_needed.update(b_replies['in_response_to_tweet_id'].astype(int))
            
        if len(parent_ids_needed) >= max_pairs * 2:
            break
            
    if not brand_replies_list:
        raise ValueError(f"No replies found for brand {brand} in {csv_path}")
        
    all_brand_replies = pd.concat(brand_replies_list, ignore_index=True)
    all_brand_replies['in_response_to_tweet_id'] = all_brand_replies['in_response_to_tweet_id'].astype(int)
    
    print(f"Found {len(all_brand_replies)} {brand} replies. Retrieving customer queries...")
    
    # Step 2: Retrieve matching customer parent tweets
    customer_tweets_list = []
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=['tweet_id', 'author_id', 'inbound', 'created_at', 'text'],
        low_memory=False
    ):
        matched = chunk[chunk['tweet_id'].isin(parent_ids_needed)]
        if not matched.empty:
            customer_tweets_list.append(matched)
            
    if not customer_tweets_list:
        raise ValueError("Could not match customer parent tweets.")
        
    all_customer_tweets = pd.concat(customer_tweets_list, ignore_index=True)
    print(f"Retrieved {len(all_customer_tweets)} matching customer parent tweets.")
    
    # Step 3: Merge customer query with brand reply
    merged = all_brand_replies.merge(
        all_customer_tweets,
        left_on='in_response_to_tweet_id',
        right_on='tweet_id',
        suffixes=('_brand', '_customer')
    )
    
    # Step 4: Clean and filter to high quality English conversations
    records = []
    for _, row in merged.iterrows():
        cust_raw = str(row['text_customer'])
        brand_raw = str(row['text_brand'])
        
        if not is_predominantly_english(cust_raw) or not is_predominantly_english(brand_raw):
            continue
            
        cust_clean = clean_text(cust_raw, is_brand_reply=False)
        brand_clean = clean_text(brand_raw, is_brand_reply=True)
        
        if len(cust_clean) < 15 or len(brand_clean) < 15:
            continue
            
        links = extract_resolution_links(brand_raw)
        
        records.append({
            'customer_tweet_id': int(row['tweet_id_customer']),
            'brand_tweet_id': int(row['tweet_id_brand']),
            'customer_text_raw': cust_raw,
            'customer_text': cust_clean,
            'brand_text_raw': brand_raw,
            'brand_text': brand_clean,
            'links': ';'.join(links),
            'brand': brand,
            'created_at_customer': row['created_at_customer'],
            'created_at_brand': row['created_at_brand']
        })
        
        if len(records) >= max_pairs:
            break
            
    df_clean = pd.DataFrame(records)
    print(f"Successfully processed {len(df_clean)} high-quality English conversation pairs.")
    return df_clean
