import re
import unicodedata
import json
from pathlib import Path


# Regular expressions for text cleaning
_whitespace_re = re.compile(r'\s+')
_punctuation_re = re.compile(r'[^\w\s]')
_numbers_re = re.compile(r'\d+')
_letters_re = re.compile(r'[a-zA-Z]+')
_abbreviations = {
    'mr.': 'mister',
    'mrs.': 'misess',
    'dr.': 'doctor',
    'no.': 'number',
    'st.': 'saint',
    'co.': 'company',
    'jr.': 'junior',
    'maj.': 'major',
    'gen.': 'general',
    'drs.': 'doctors',
    'rev.': 'reverend',
    'lt.': 'lieutenant',
    'hon.': 'honorable',
    'sgt.': 'sergeant',
    'capt.': 'captain',
    'esq.': 'esquire',
    'ltd.': 'limited',
    'col.': 'colonel',
    'ft.': 'fort',
}


def normalize_text(text):
    """
    Normalize text (lowercase, remove punctuation, etc.)
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Replace abbreviations
    for abbr, expansion in _abbreviations.items():
        text = text.replace(abbr, expansion)
    
    # Remove punctuation
    text = re.sub(_punctuation_re, ' ', text)
    
    # Collapse whitespace
    text = re.sub(_whitespace_re, ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def english_cleaners(text):
    """
    Pipeline for English text cleaning
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    
    # Convert numbers to words
    text = re.sub(_numbers_re, _number_to_words, text)
    
    # Apply general normalization
    text = normalize_text(text)
    
    return text


def _number_to_words(match):
    """
    Convert number to words
    
    Args:
        match: Regex match object
        
    Returns:
        Number as words
    """
    try:
        import inflect
        p = inflect.engine()
        return p.number_to_words(match.group(0))
    except ImportError:
        # If inflect is not installed, return the original number
        return match.group(0)


def text_to_phonemes(text, g2p_model="espeak", cache=None):
    """
    This function is kept for compatibility but now raises an error.
    The system should only use phonemes directly from .lab files.
    """
    raise NotImplementedError(
        "text_to_phonemes is no longer supported. "
        "The system should only use phonemes directly from .lab files."
    )


def load_phoneme_dictionary(path):
    """
    Load phoneme dictionary from file
    
    Args:
        path: Path to dictionary file
        
    Returns:
        Phoneme dictionary
    """
    with open(path, 'r') as f:
        phoneme_dict = json.load(f)
    
    return phoneme_dict


def save_phoneme_dictionary(phoneme_dict, path):
    """
    Save phoneme dictionary to file
    
    Args:
        phoneme_dict: Phoneme dictionary
        path: Path to save dictionary
    """
    with open(path, 'w') as f:
        json.dump(phoneme_dict, f, indent=2)


def build_phoneme_vocabulary(texts, g2p_model="espeak"):
    """
    Build phoneme vocabulary from texts
    
    Args:
        texts: List of texts
        g2p_model: Grapheme-to-phoneme model name
        
    Returns:
        Phoneme vocabulary dictionary
    """
    # Initialize vocabulary with special tokens
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    
    # Process texts
    phoneme_set = set()
    for text in texts:
        phonemes = text_to_phonemes(text, g2p_model)
        phoneme_set.update(phonemes)
    
    # Add phonemes to vocabulary
    for i, phoneme in enumerate(sorted(phoneme_set)):
        vocab[phoneme] = i + 4
    
    return vocab