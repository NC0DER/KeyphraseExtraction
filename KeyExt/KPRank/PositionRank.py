#! /usr/bin/env python
# -*- coding: utf-8 -*-

from doc_candidates import LoadFile
import networkx as nx
from numpy import dot
from numpy.linalg import norm
import numpy as np
from math import log10
from collections import defaultdict
import operator
import unicodedata

def normalize_text(text):
    if not isinstance(text, unicode):
        text = unicode(text, 'utf-8')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore') 
    return text
    
class PositionRank(LoadFile):

    def __init__(self, input_text, window, phrase_type, emb_dim, embeddings):
        """ Redefining initializer for PositionRank. """

        super(PositionRank, self).__init__(input_text=input_text)

        self.graph = nx.Graph()
        """ The word graph. """
        self.window = window

        self.phrase_type = phrase_type
        self.emb_dim = emb_dim
        self.embeddings = embeddings#KeyedVectors.load_word2vec_format(emb_file, binary=True)
        self.random_embeddings = {}
    
    
    def get_cosine_dist(self, word1, word2):
        curr_embeddings1 = []
        if word1.lower() in self.embeddings:
            curr_embeddings1 = self.embeddings[word1.lower()] 
        elif word1.lower() in self.random_embeddings:
            curr_embeddings1 = self.random_embeddings[word1.lower()]
        else:
            curr_embeddings1 = np.random.rand(self.emb_dim)
            self.random_embeddings[word1.lower()] = curr_embeddings1
            
        curr_embeddings2 = []
        if word2.lower() in self.embeddings:
            curr_embeddings2 = self.embeddings[word2.lower()] 
        elif word2.lower() in self.random_embeddings:
            curr_embeddings2 = self.random_embeddings[word2.lower()]
        else:
            curr_embeddings2 = np.random.rand(self.emb_dim)
            self.random_embeddings[word2.lower()] = curr_embeddings2
        
        cos_sim = 0.0
        if (norm(curr_embeddings1)*norm(curr_embeddings2)) != 0:
            #print curr_embeddings1
            #print curr_embeddings2
            cos_sim = dot(curr_embeddings1, curr_embeddings2)/(norm(curr_embeddings1)*norm(curr_embeddings2))
        semantic_val = 0.0
        if cos_sim != 1.0:
            semantic_val = 1.0 / (1.0 - cos_sim)
            
        return semantic_val


    def build_graph(self, window, pos=None):
        """
        build the word graph

        :param window: the size of window to add edges in the graph
        :param pos: he part of speech tags used to select the graph's nodes
        :return:
        """

        if pos is None:
            pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

        # container for the nodes
        seq = []        
        individual_count = {} # my addition
        stemmed_original_map = {}
        
        # select nodes to be added in the graph
        for el in self.words:
            if el.pos_pattern in pos:
                seq.append((el.stemmed_form, el.position, el.sentence_id))
                self.graph.add_node(el.stemmed_form)
                if el.stemmed_form not in individual_count:
                    individual_count[el.stemmed_form] = 0
                individual_count[el.stemmed_form] += 1
                if el.stemmed_form not in stemmed_original_map:
                    stemmed_original_map[el.stemmed_form] = el.surface_form
        
        # add edges
        for i in range(0, len(seq)):
            for j in range(i+1, len(seq)):
                if seq[i][1] != seq[j][1] and abs(j-i) < window:
                    if not self.graph.has_edge(seq[i][0], seq[j][0]):
                        self.graph.add_edge(seq[i][0], seq[j][0], weight=1)
                    else:
                        self.graph[seq[i][0]][seq[j][0]]['weight'] += 1
        
    def candidate_selection(self, pos=None, phrase_type='n_grams'):
        """
        the candidates selection for PositionRank
        :param pos: pos: the part of speech tags used to select candidates
        :return:
        """

        if pos is None:
            pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

        # uncomment the line below if you wish to extract ngrams instead of the longest phrase
        if phrase_type=='n_grams':
            self.get_ngrams(n=4, good_pos=pos)
        else:
            # select the longest phrase as candidate keyphrases
            self.get_phrases(self, good_pos=pos)


    def candidate_scoring(self, pos=None, window=10, theme_mode = 'adj_noun_title' ,update_scoring_method=False):
        """
        compute a score for each candidate based on PageRank algorithm
        :param pos: the part of speech tags
        :param window: window size
        :param update_scoring_method: if you want to update the scoring method based on my paper cited below:
        Florescu, Corina, and Cornelia Caragea. "A New Scheme for Scoring Phrases in Unsupervised Keyphrase Extraction."
         European Conference on Information Retrieval. Springer, Cham, 2017.

        :return:
        """
        
        if pos is None:
            pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

        # build the word graph
        self.build_graph(window=window, pos=pos)

        # filter out canditates that unlikely to be keyphrases
        self.filter_candidates(max_phrase_length=4, min_word_length=3, valid_punctuation='-.')

        ######### get Theme scores ########

        # get the theme vector
        theme_vec = np.array([0] * self.emb_dim)
        
        if theme_mode == 'adj_noun_title':
            tv_words = 0
            for w, p in self.sentences[0]:
                w = w.lower()
                if p in pos:
                    if w in self.embeddings['words']: # Fix embeddings structure bug.
                        curr_vec = np.array(self.embeddings['embeddings'][self.embeddings['words'].index(w)])
                        theme_vec = theme_vec + curr_vec    
                        tv_words += 1
            if tv_words > 0:
                theme_vec = theme_vec / tv_words
                        
        elif theme_mode == 'adj_noun_all':
            tv_words = 0
            for sentence in self.sentences:
                for w, p in sentence:
                    w = w.lower()
                    if p in pos:
                        if w in self.embeddings['words']: # Fix embeddings structure bug.
                            curr_vec = np.array(self.embeddings['embeddings'][self.embeddings['words'].index(w)])
                            theme_vec = theme_vec + curr_vec           
                            tv_words += 1
            if tv_words > 0:        
                theme_vec = theme_vec / tv_words
            
        elif theme_mode == 'cls_title':
            theme_vec = self.embeddings['cls_ttl']
        elif theme_mode == 'cls_all':
            theme_vec = self.embeddings['cls_all']
        elif theme_mode == 'mean_title':
            theme_vec = self.embeddings['mean_ttl']
        elif theme_mode == 'mean_all':
            theme_vec = self.embeddings['mean_all']
            
        # get the thematic scores            
        personalization_k2v = {}
        for w in self.words:
            word = w.surface_form
            stem = w.stemmed_form
            curr_pos = w.pos_pattern
            word = word.lower()
            if curr_pos in pos:
                if stem not in personalization_k2v.keys():
                    curr_vec = []
                    if word in self.embeddings['words']: # Fix embeddings structure bug.
                        print(theme_mode + ': EMB-FOUND')
                        curr_vec = self.embeddings['embeddings'][self.embeddings['words'].index(word)]
                    elif word in self.random_embeddings:
                        curr_vec = self.random_embeddings[word]
                    else:
                        curr_vec = np.random.rand(self.emb_dim)
                        self.random_embeddings[word] = curr_vec
                        print('EMB-NOT-FOUND')
                    cos_sim = 0.000000001
                    if (norm(curr_vec)*norm(theme_vec)) != 0.0:
                        cos_sim = dot(curr_vec, theme_vec)/(norm(curr_vec)*norm(theme_vec))
                    personalization_k2v[stem] = cos_sim

        ######### get Positional scores ########
        personalization_pr = {}
        for w in self.words:
            stem = w.stemmed_form
            poz = w.position
            pos = w.pos_pattern

            if pos in pos:
                if stem not in personalization_pr:
                    personalization_pr[stem] = 1.0/poz
                else:
                    personalization_pr[stem] = personalization_pr[stem]+1.0/poz

        ######## multiply both scores #######
        ipdict=[personalization_k2v, personalization_pr]

        output=defaultdict(lambda:1)
        for d in ipdict:
            for item in d:
               output[item] *= d[item]
        
        personalization = dict(output)
        
        ######## normalize scores ########        
        factor = 1.0 / sum(personalization.values())

        normalized_personalization = {k: v * factor for k, v in personalization.items()}
        
        # compute the word scores using personalized random walk
        pagerank_weights = nx.pagerank_scipy(self.graph, personalization=normalized_personalization, weight='weight')
        #pagerank_weights = normalized_personalization

        
        # loop through the candidates
        if update_scoring_method:
            for c in self.candidates:
                if len(c.stemmed_form.split()) > 1:
                    # for arithmetic mean
                    #self.weights[c.stemmed_form] = [stem.stemmed_form for stem in self.candidates].count(c.stemmed_form) * \
                                                   #sum([pagerank_weights[t] for t in c.stemmed_form.split()]) \
                                                   #/ len(c.stemmed_form.split())
                    # for harmonic mean
                    self.weights[c.stemmed_form] = [cand.stemmed_form for cand in self.candidates].count(c.stemmed_form) * \
                                                   len(c.stemmed_form.split()) / sum([1.0 / pagerank_weights[t] for t in c.stemmed_form.split()])
                else:
                    self.weights[c.stemmed_form] = pagerank_weights[c.stemmed_form]
        else:
            for c in self.candidates:
                self.weights[c.stemmed_form] = sum([pagerank_weights[t] for t in c.stemmed_form.split()])



