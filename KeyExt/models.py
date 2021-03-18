import pke
import pytextrank
from string import printable
from statistics import mean
from operator import itemgetter
from itertools import islice, combinations
from nltk import sent_tokenize
from RAKE import Rake, NLTKStopList
from yake import KeywordExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from KeyExt.utils import counter

@counter
def tfidfvectorizer(text, ngram_range = (1, 3), top_n = 10):

    # Tokenize the text into sentences.
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer (
        stop_words = 'english', 
        ngram_range = ngram_range
    )
    # Vectorizer fits and transform the sentences.
    vectorizer.fit_transform(sentences)
    results = {
        key: val 
        for key, val in sorted (
            vectorizer.vocabulary_.items(), 
            key = lambda item: item[1],
            reverse = True
        )
    }
    return list(islice(results, top_n))

@counter
def keybert(text, model, ngram_range = (1, 3), top_n = 10, measure = None, diversity = 0.5):

    # Returned the extracted keywords based on the specified arguments. 
    return [
        keyphrase for (keyphrase, _) in 
        model.extract_keywords (
            text, 
            keyphrase_ngram_range = ngram_range,
            stop_words = 'english',
            top_n = top_n,
            nr_candidates = 2 * top_n,
            use_maxsum = True if measure == 'maxsum' else False,
            use_mmr = True if measure == 'mmr' else False,
            diversity = diversity
    )]

@counter
def textrank(text, nlp, top_n = 10):

    # Temporarily add PyTextRank to the spaCy pipeline.
    nlp.add_pipe('textrank', last = True)

    # Perform nlp on text.
    doc = nlp(text)

    # Remove textrank from the pipeline.
    nlp.remove_pipe('textrank')

    # Return the top N phrases from the document.
    return [phrase.text for phrase in doc._.phrases][:top_n]

@counter
def singlerank(text, top_n = 10):

    # Clean the text from non-printable characters.
    text = ''.join(word for word in text if word in printable)

    # Initialize the keyphrase extraction model.
    extractor = pke.unsupervised.SingleRank()

    # Load the content of the document and preprocess it with spacy.
    # Then, select the keyphrase candidates from the document,
    # and weight them using a random walk algorithm.
    extractor.load_document(input = text, language = 'en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    
    # Return the n-highest scored candidates.
    return [
        keyphrase for (keyphrase, score)
        in extractor.get_n_best(n = top_n, redundancy_removal = True)
    ]

@counter
def rake(text, top_n = 10):

    # Clean the text from non-printable characters.
    text = ''.join(word for word in text if word in printable)

    # Uses all english stopwords and punctuation from NLTK.
    r = Rake(NLTKStopList())
    return [keyphrase for (keyphrase, score) in r.run(text)[:top_n]]

@counter
def yake(text, top_n = 10, n = 3, dedupLim = 0.9, dedupFunc = 'seqm', windowsSize = 1):

    # Initialize the keyword extractor object and its parameters.
    kw_extractor = KeywordExtractor (
        top = top_n,
        n = n,
        dedupLim = dedupLim,
        dedupFunc = dedupFunc,
        windowsSize = windowsSize
    )
    # Return the extracted keywords, in a list.
    return [keyword for (keyword, score) in kw_extractor.extract_keywords(text)]
