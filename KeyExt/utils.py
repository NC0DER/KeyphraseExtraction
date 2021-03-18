import os
import time
import spacy
import platform
import functools
import KeyExt.config
from keybert import KeyBERT
from string import punctuation
from nltk.stem import SnowballStemmer
from stempel import StempelStemmer


# Initialize all required stemmers once.
stemmers = {
    'english': SnowballStemmer('english'),
    'french': SnowballStemmer('french'),
    'spanish': SnowballStemmer('spanish'),
    'portuguese': SnowballStemmer('portuguese'),
    'polish': StempelStemmer.default()
}

def load_models():
    """
    Function which loads the english NLP model, and the Keybert model.
    This needs to run once since all models need a few seconds to load.
    """
    return (
        spacy.load('en_core_web_sm'),
        KeyBERT('distiluse-base-multilingual-cased-v2')
    )

def preprocess(lis, language):
    """
    Function which applies stemming to a 
    lowercase version of each string of the list,
    which has all punctuation removed.
    """
    return list(map(stemmers[language].stem, 
           map(lambda s: s.translate(str.maketrans('', '', punctuation)),
           map(str.lower, lis))))


def rreplace(s, old, new, occurrence):
    """
    Function which replaces a string occurence
    in a string from the end of the string.
    """
    return new.join(s.rsplit(old, occurrence))

def clear_screen():
    """
    Function which clears the output of the terminal 
    by using the platform specific system call.
    """
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear') # Linux/OS X.
    return

def counter(func):
    """
    Print the elapsed system time in seconds, 
    if only the debug flag is set to True.
    """
    if not KeyExt.config.debug:
        return func
    @functools.wraps(func)
    def wrapper_counter(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'{func.__name__}: {end_time - start_time} secs')
        return result
    return wrapper_counter
