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
    Convert text to phoneme sequence
    
    Args:
        text: Input text
        g2p_model: Grapheme-to-phoneme model name
        cache: Path to cache file (optional)
        
    Returns:
        List of phonemes
    """
    # Load cache if provided
    phoneme_cache = {}
    if cache is not None:
        cache_path = Path(cache)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                phoneme_cache = json.load(f)
            
            # Check if text is in cache
            if text in phoneme_cache:
                return phoneme_cache[text].split()
    
    # Clean text
    cleaned_text = english_cleaners(text)
    
    # Apply G2P model
    if g2p_model == "espeak":
        try:
            import phonemizer
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator
            
            # Initialize phonemizer
            backend = EspeakBackend(
                language='en-us',
                separator=Separator(phone=' ', syllable=None, word=None)
            )
            
            # Convert text to phonemes
            phonemes = backend.phonemize([cleaned_text], strip=True)[0].split()
            
            # Update cache
            if cache is not None:
                phoneme_cache[text] = ' '.join(phonemes)
                with open(cache_path, 'w') as f:
                    json.dump(phoneme_cache, f, indent=2)
            
            return phonemes
        except ImportError:
            raise ImportError("Please install phonemizer: pip install phonemizer")
    else:
        raise ValueError(f"Unsupported G2P model: {g2p_model}")


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