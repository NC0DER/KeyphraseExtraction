import os
import time
import platform
import functools
import KeyExt.config
from string import punctuation
from nltk.stem import SnowballStemmer


# Initialize the English stemmer once.
stemmer = SnowballStemmer('english')


def preprocess(lis):
    """
    Function which applies stemming to a 
    lowercase version of each string of the list,
    which has all punctuation removed.
    """
    return list(map(stemmer.stem, 
           map(lambda s: s.translate(str.maketrans('', '', punctuation)),
           map(str.lower, lis))))


def rreplace(s, old, new, occurrence):
    """
    Function which replaces a string occurence
    in a string from the end of the string.
    """
    return new.join(s.rsplit(old, occurrence))
