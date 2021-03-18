import json
import KeyExt.models
from KeyExt.experiments import run_experiments
from KeyExt.utils import load_models
from KeyExt.config import input_path, output_path, top_n

def keyword_extraction():

    text = """Machine learning (ML) is the study of computer algorithms 
    that improve automatically through experience.
    It is seen as a part of artificial intelligence. 
    Machine learning algorithms build a model based on sample data, 
    known as "training data", in order to make predictions or decisions 
    without being explicitly programmed to do so. Machine learning algorithms 
    are used in a wide variety of applications, such as email filtering and computer vision, 
    where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
    A subset of machine learning is closely related to computational statistics, 
    which focuses on making predictions using computers; 
    but not all machine learning is statistical learning. 
    The study of mathematical optimization delivers methods, theory and application domains 
    to the field of machine learning. Data mining is a related field of study, 
    focusing on exploratory data analysis through unsupervised learning. 
    In its application across business problems, machine learning 
    is also referred to as predictive analytics."""
   
    # Initialize the spacy and keybert models.
    nlp_model, bert_model = load_models()

    # Create all ngrams with range (1, 3) for the text.
    ngrams = {
        '1-tfidfvectorizer': KeyExt.models.tfidfvectorizer(text, top_n = 10),
        '2-keybert-maxsum': KeyExt.models.keybert(text, bert_model, top_n = 10, measure = 'maxsum', diversity = 0.7),
        '3-keybert-mmr': KeyExt.models.keybert(text, bert_model, top_n = 10, measure = 'mmr', diversity = 0.7),
        '4-textrank': KeyExt.models.textrank(text, nlp_model, top_n = 10),
        '5-singlerank': KeyExt.models.singlerank(text, top_n = 10),
        '6-rake': KeyExt.models.rake(text, top_n = 10),
        '7-yake-seqm': KeyExt.models.yake(text, top_n = 10, dedupFunc = 'seqm'),
    }

    # Write ngrams from each method to a json file.
    with open(r'C:\Users\Nick\Desktop\ngrams.json', 'w',
        encoding = 'utf-8-sig', errors = 'ignore') as file:
        file.write(json.dumps(ngrams, indent = 4, separators = (',', ':')))

if __name__=='__main__': run_experiments(input_path, output_path, top_n)
