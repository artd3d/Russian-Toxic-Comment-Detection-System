"""
Shared tokenizer module for Russian Toxic Comment Detection
This module contains the RussianTokenizer class that can be imported and pickled properly
"""

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from typing import List

def initialize_nltk():
    """Download and initialize required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("NLTK data downloaded successfully!")

# Initialize NLTK data
initialize_nltk()

class RussianTokenizer:
    """Custom tokenizer class that can be pickled properly"""
    
    def __init__(self, remove_stop_words=True):
        self.remove_stop_words = remove_stop_words
        self.snowball = SnowballStemmer(language="russian")
        self.russian_stop_words = stopwords.words('russian')
    
    def __call__(self, sentence: str) -> List[str]:
        """Tokenize and preprocess Russian text"""
        try:
            tokens = word_tokenize(sentence, language="russian")
        except LookupError:
            print("Warning: Using English tokenizer as fallback")
            tokens = word_tokenize(sentence, language="english")
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove Russian stopwords if requested
        if self.remove_stop_words:
            tokens = [token for token in tokens if token not in self.russian_stop_words]
        
        # Apply stemming
        tokens = [self.snowball.stem(token) for token in tokens]
        
        return tokens

def tokenaiz_sentens(sentence: str, remove_stop_words: bool = True) -> List[str]:
    """
    Tokenize and preprocess Russian text (legacy function for compatibility)
    
    Args:
        sentence (str): Input Russian text
        remove_stop_words (bool): Whether to remove Russian stop words
        
    Returns:
        List[str]: List of preprocessed tokens
    """
    tokenizer = RussianTokenizer(remove_stop_words)
    return tokenizer(sentence)