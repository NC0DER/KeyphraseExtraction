#! /usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import itertools
from nltk.stem.porter import PorterStemmer
import os.path
from nltk import word_tokenize
from string import punctuation
import re
import unicodedata


def read_input_file(this_file):
    # read the text of the file; if the file cannot be read then the file is excluded
    if os.path.exists(this_file):
        with codecs.open(this_file, "r", encoding='utf-8') as f:
            #text = f.read()
            lines = f.readlines()
            lines[0] = lines[0].strip()
            if not (lines[0].endswith(".") or lines[0].endswith("?") or lines[0].endswith("!")):
                lines[0] = lines[0]+'.'
            text = ' '.join(lines)
        f.close()
    else:
        text = None

    return text


def read_gold_file(this_gold):

    # read the gold file; if the file cannot be read (does not exist) the file is excluded
    if os.path.exists(this_gold):
        with codecs.open(this_gold, "r", encoding='utf-8') as f:
            gold_list = f.readlines()
        f.close()
    else:
        gold_list = None

    return gold_list

def get_ascii(text):
    if not isinstance(text, unicode):
            text = unicode(text, "utf-8")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
    return text 
        

def get_stemmed_words_and_stemmed_text(text):
    stemmer = PorterStemmer()
    text_words = text.split()
    text_words_stem = []
    for word in text_words:
        text_words_stem.append(stemmer.stem(word))
    text_stem = ' '.join(text_words_stem)
    return text_words_stem, text_stem
        

def load_stemmed_gold_phrases(lines):
    punct_list = ['\'', '"', '\\', '!', '@', '#', '$', '%', 
              '^', '&', '*', '(', ')', '_', '-', '+', '=','{', '}', '[', ']', 
              '|', ':', ';', '<', '>', ',', '.', '?', '/', '`', '~']

    punct_re = '|'.join(map(re.escape, punct_list))
    
    gold_phrases = []
    for line in lines:
        line = line.strip()
        line = line.lower()
        line = get_ascii(line)
        line = re.sub(punct_re, ' ', line)
        line = re.sub('\s+', ' ', line).strip()
        line_words_stem, line_stem = get_stemmed_words_and_stemmed_text(line)
        gold_phrases.append(line_stem)
    return gold_phrases

def tokenize(text, encoding):
    """ tokenize text
    Args:
        text: tect to be tokenized
        """
    return [token for token in word_tokenize(text.lower().decode(encoding))]


def filter_candidates(tokens, stopwords_file=None, min_word_length=2, valid_punctuation='-'):
    """ discard candidates based on various criteria
    Args:
        tokens: tokens to be filtered out
        stopwords_file: if you want to load a file with stopwords
        min_word_length: filter words shorter than min_word_length
        valid_punctuation: filter words that contain other punctuation than valid_punctuation
        encoding='utf-8'
        """

    # if a list of stopwords is not provided then load the stopwords'list from nltk
    stopwords_list = []
    if stopwords_file is None:
        from nltk.corpus import stopwords
        stopwords_list = set(stopwords.words('english'))
    else:
        with codecs.open(stopwords_file, 'rb', encoding='utf-8') as f:
            f.readlines()
        f.close()
        # add the stopword from file in the stopwords_list container
        for line in f:
            stopwords_list.append(line)

    # keep indices to be deleted
    indices = []

    for i, c in enumerate(tokens):

        # discard those candidates that contain stopwords
        if c in stopwords_list:
            indices.append(i)

        # discard candidates that contain words shorter that min_word_length
        elif len(c) < min_word_length:
            indices.append(i)

        elif c in ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']:
            indices.append(i)

        else:

            # discard candidates that contain other characters except letter, digits, and valid punctuation
            letters_set = set([u for u in c])

            if letters_set.issubset(punctuation):
                indices.append(i)

            elif re.match(r'^[a-zA-Z0-9%s]*$' % valid_punctuation, c):
                pass
            else:
                indices.append(i)

    dels = 0

    for index in indices:
        offset = index - dels
        del tokens[offset]
        dels += 1

    return tokens


def stemming(text):
    """ stem tokens """
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(i) for i in text]


def iter_data(path_to_data, encoding):
    """Yield each article from the Medline """
    files = []
    #with open('/home/corina/Documents/Research/Projects/unsupervisedKE/data_analysis/medline_10000_1.txt','rb') as rf:
        #filenames = rf.readlines()
        #files = [file.strip() for file in filenames]
    #rf.close()
    #print files
    i=1
    #for filename in filenames: #os.listdir(path_to_data):
    for filename in os.listdir(path_to_data):
        #filename = filename.strip()

        i += 1
        with open(path_to_data + filename, 'rb') as f:
            text = f.read().strip()
            tokens = tokenize(text, encoding)
            tokens = filter_candidates(tokens)
            tokens = stemming(tokens)
        f.close()
        yield path_to_data + filename, text, tokens


class MyCorpus(object):

    def __init__(self, path_to_data, dictionary, length=None, encoding='utf-8'):
        """
        Parse the collection of documents from file path_to_data.
        Yield each document in turn, as a list of tokens.
        Args:
            path_to_data: the location of the collection
            dictionary: the mapping between word and ids
            length: the number of docs in the corpus
        """
        self.path_to_data = path_to_data
        self.dictionary = dictionary
        self.length = length
        self.encoding = encoding
        self.index_filename = {}

    def __iter__(self):

        index = 0

        for filename, text, tokens in itertools.islice(iter_data(self.path_to_data, self.encoding), self.length):
            self.index_filename[index] = filename
            index += 1
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        if self.length is None:
            self.length = sum(1 for doc in self)
        return self.length
