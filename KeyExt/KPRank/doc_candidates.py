#! /usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
from nltk import pos_tag, word_tokenize, sent_tokenize
import re
from nltk.stem.porter import PorterStemmer
from string import punctuation
import operator
porter_stemmer = PorterStemmer()


class Candidate(object):
    """ The data structure for candidates. """

    def __init__(self, surface_form, pos_pattern, stemmed_form, position, sentence_id):
        """

        :param surface_form: the token form as it appears in the text
        :param pos_pattern: the sequence of pos-tags for the candidate
        :param stemmed_form: the stemmed form of the candidate keyphrase
        :param position: its current position in the document
        :param sentence_id: the number of sentence the candidate appears in.
        """

        self.surface_form = surface_form
        """the candidate form occuring in the text"""

        self.pos_pattern = pos_pattern
        """ the part-of-speech of the candidate. """

        self.stemmed_form = stemmed_form
        """ the stemmed form of the candidate. """

        self.position = position
        """ those positions in the document where the candidate occurs. """

        self.sentence_id = sentence_id
        """ the number of the sentence in the document where the candidate occurs. """


class LoadFile(object):
    """ The LoadFile class that provides base functions. """

    def __init__(self, input_text):
        """
        Initializer for LoadFile class
        :param input_text: the text of the input file
        """

        self.input_text = input_text
        """ The path of the input file. """

        self.sentences = []
        """ The sentence of the input file. """

        self.stemmed_sentences = []
        """ The sentence of the input file processed based on the user needs (lower case, stemmed, stopwords). """

        self.words = []
        """ The individual words of the document (unigrams)"""
       
        self.candidates = []
        """ The candidates of the input file. """

        self.weights = {}
        """ The weight/score of candidates. """

        self.gold_keyphrases = []
        
        punct_list = ['\'', '"', '\\', '!', '@', '#', '$', '%', 
              '^', '&', '*', '(', ')', '_', '-', '+', '=','{', '}', '[', ']', 
              '|', ':', ';', '<', '>', ',', '.', '?', '/', '`', '~']

        self.punct_re = '|'.join(map(re.escape, punct_list))

        # the text is tokenized at the sentence level and part-of-speech tags are assigned
        self.sentences = [pos_tag(word_tokenize(s)) for s in sent_tokenize(self.input_text.lower())]
        #self.sentences = [pos_tag(re.sub(self.punct_re, ' ', s).strip().split()) for s in sent_tokenize(self.input_text.lower())]

        # a stemmed form of sentences is also saved
        for s in self.sentences:
            self.stemmed_sentences.append([porter_stemmer.stem(w) for w, p in s])

    def get_doc_words(self):

        """
        extracts all of the individual words from the document
        we save all words regarless of the part-of-speech tags
        for each word we get its surface form, pos-tag and stemmed form

        """
        tokens = []
        pos_tags = []
        stems = []

        # get a list of document's words and another one of its part-of-speech tags
        for i in range(0, len(self.sentences)):
            tokens.extend([(w, i) for w, p in self.sentences[i]])
            pos_tags.extend([p for w, p in self.sentences[i]])

        # get the stems  of the words as another list
        for s in self.stemmed_sentences:
            stems.extend([w for w in s])

        # add the word in the words container
        for i in range(0, len(tokens)):
            self.words.append(Candidate(tokens[i][0], pos_tags[i], stems[i], (i+1), tokens[i][1]))

    def get_ngrams(self, n=3, good_pos=None):
        """
        compute all the ngrams or ngrams with patters found in the document

        :param n: the maximum length of the ngram (default is 3)
        :param good_pos: goodPOS if any word filter is applied, for eg. keep only nouns and adjectives
        :return:
        """

        jump = 0
        if good_pos:
            for i in range(0, len(self.sentences)):

                # jump helps to keep track of a word position; its leap is the sentence length
                if i == 0:
                    jump = 0
                else:
                    jump += len(self.sentences[i-1])

                # if the sentence is very short then the maximum length of the phrase is the sentence's length
                max_length = min(n, len(self.sentences[i]))

                tokens = [w for w, p in self.sentences[i]]
                pos_tags = [p for w, p in self.sentences[i]]
                stems = self.stemmed_sentences[i]
                for j in range(0, len(tokens)):
                    for k in range(j, len(tokens)):
                        if pos_tags[k] in good_pos and k-j < max_length and k < (len(tokens)-1):
                            self.candidates.append(Candidate(' '.join(tokens[j:k+1]), ' '.join(pos_tags[j:k+1]),
                                                             ' '.join(stems[j:k+1]), j+jump, i))
                        else:
                            break
        else:
            for i in range(0, len(self.sentences)):

                # jump helps to keep track of a word position; its leap is the sentence length
                if i == 0:
                    jump = 0
                else:
                    jump += len(self.sentences[i-1])

                # if the sentence is very short then the maximum length of the phrase is the sentence's length
                max_length = min(n, len(self.sentences[i]))

                tokens = [w for w, p in self.sentences[i]]
                pos_tags = [p for w, p in self.sentences[i]]
                stems = self.stemmed_sentences[i]
                for j in range(0, len(tokens)):
                    for k in range(j, len(tokens)):
                        if k-j < max_length and k < (len(tokens)-1):
                            self.candidates.append(Candidate(' '.join(tokens[j:k+1]), ' '.join(pos_tags[j:k+1]),
                                                             ' '.join(stems[j:k+1]), j+jump, i))
                        else:
                            break

    def get_phrases(self, n=4, good_pos=None):
        """
        extract the longest phrase

        :param n: the maximum length of a phrase
        :param good_pos: a sequence of pos tags based on which the phrases are extracted
        :return:
        """

        if good_pos is None:
            good_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

        jump = 0
        for i in range(0, len(self.sentences)):

            # jump helps to keep track of a word position; its leap is the sentence length
            if i == 0:
                jump = 0
            else:
                jump += len(self.sentences[i-1])

            # if the sentence is very short then the maximum length of the phrase is the sentence's length
            max_length = min(n, len(self.sentences[i]))

            tokens = [w for w, p in self.sentences[i]]
            pos_tags = [p for w, p in self.sentences[i]]
            stems = self.stemmed_sentences[i]
            j = 0
            while j < len(tokens):
                for k in range(j, len(tokens)):
                    if pos_tags[k] in good_pos and k - j < max_length and k < (len(tokens)-1):
                        continue
                    else:
                        if j < k:
                            self.candidates.append(
                                Candidate(' '.join(tokens[j:k]), ' '.join(pos_tags[j:k]), ' '.join(stems[j:k]),
                                          j+jump, i))
                        j = k+1
                        break

    def filter_candidates(self, stopwords_file=None, max_phrase_length=4, min_word_length=3, valid_punctuation='-.'):
        """
         discard candidates based on various criteria

        :param stopwords_file: a stop-word file that the user wants to input;
        :param max_phrase_length: filter out phrases longer than max_phrase_length
        :param min_word_length:  filter out phrases that contain words shorter than min_word_length
        :param valid_punctuation: keep tokens that contain any of the valid punctuation
        :return:
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

            for line in f:
                stopwords_list.append(line)

        indices = []
        for i, c in enumerate(self.candidates):

            tokens = c.surface_form.split()
            pos = c.pos_pattern.split()

            # discard those candidates that contain stopwords
            if set(tokens).intersection(stopwords_list):
                indices.append(i)

            # discard candidates longer than max_phrase_length
            elif len(tokens) > max_phrase_length:
                indices.append(i)

            # discard candidates that contain words shorter that min_word_length
            elif min([len(t) for t in tokens]) < min_word_length:
                indices.append(i)

            # discard candidates that end in adjectives (including single word adjectives)
            elif pos[-1] == 'JJ':
                indices.append(i)

            elif set(tokens).intersection(set(['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'])):
                indices.append(i)

            else:

                # discard candidates that contain other characters except letter, digits, and valid punctuation
                for word in tokens:
                    letters_set = set([u for u in word])

                    if letters_set.issubset(punctuation):
                        indices.append(i)
                        break

                    elif re.match(r'^[a-zA-Z0-9%s]*$' % valid_punctuation, word):
                        continue

                    else:
                        indices.append(i)
                        break

        dels = 0
        for index in indices:
            offset = index - dels
            dels += 1
            del self.candidates[offset]

    def get_best_k(self, k=10):
        """
        return top k predicted keyphrases for the current document
        :param k: top keyphrases to be retuned
        :return: top k keyphrases and their weights
        """

        # sort the candidates in reverse order based on their weights
        sorted_weights = sorted(self.weights, key=self.weights.get, reverse=True)

        # return only the k keyphrases
        return sorted_weights[:(min(k, len(sorted_weights)))]
        
    def get_best_k_with_scores(self, k=10):
        """
        return top k predicted keyphrases for the current document
        :param k: top keyphrases to be retuned
        :return: top k keyphrases and their weights
        """

        # sort the candidates in reverse order based on their weights
        sorted_weights = sorted(self.weights.items(), key=operator.itemgetter(1),reverse=True)

        # return only the k keyphrases
        return sorted_weights[:(min(k, len(sorted_weights)))]


def get_phrases_extra(text, n=4, good_pos=None):
    """
    used in ExpandRank to create phrase from neighboring documents
    :param text: text of the neighbour documents
    :param n: the lenghth of the phrase to be extracted
    :param good_pos: the set of pos tags to be considered
    :return: phrases of that neighbor document
    """
    phrases = []

    stemmed_sentences = []

    sentences = [pos_tag(word_tokenize(s)) for s in sent_tokenize(text.lower())]

    # a stemmed form of sentences is also saved
    for s in sentences:
        stemmed_sentences.append([porter_stemmer.stem(w) for w, p in s])

    if good_pos is None:
        good_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

    jump = 0
    for i in range(0, len(sentences)):

        # jump helps to keep track of a word position; its leap is the sentence length
        if i == 0:
            jump = 0
        else:
            jump += len(sentences[i-1])

        # if the sentence is very short then the maximum length of the phrase is the sentence's length
        max_length = min(n, len(sentences[i]))

        tokens = [w for w, p in sentences[i]]
        pos_tags = [p for w, p in sentences[i]]
        stems = stemmed_sentences[i]
        j = 0
        while j < len(tokens):
            for k in range(j, len(tokens)):
                if pos_tags[k] in good_pos and k - j < max_length and k < (len(tokens)-1):
                    continue
                else:
                    if j < k:
                        phrases.append(
                            Candidate(' '.join(tokens[j:k]), ' '.join(pos_tags[j:k]), ' '.join(stems[j:k]),
                                      j+jump, i))
                    j = k+1
                    break
    return phrases


