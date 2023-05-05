import os
import pke
import time
import yake
import spacy
import string
import pathlib
import functools
import pytextrank

def counter(func):
    """
    Print the elapsed system time in seconds.
    """
    @functools.wraps(func)
    def wrapper_counter(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'{func.__name__}: {end_time - start_time} secs')
        return result
    return wrapper_counter

@counter
def kpminer(text, top_n = 10):
     weights_file = r'..\pke\models\df-semeval2010.tsv.gz'
     extractor = pke.unsupervised.KPMiner()
     extractor.load_document(input = text, language = 'en')
     extractor.candidate_selection(lasf = 5, cutoff = 200)
     df = pke.load_document_frequency_file(input_file = weights_file)
     extractor.candidate_weighting(df = df, alpha = 2.3, sigma = 3.0)
     keyphrases = [key for key,_ in extractor.get_n_best(n = top_n)]
     return keyphrases

@counter
def mprank(text, top_n = 10):
    extractor = pke.unsupervised.MultipartiteRank()
    stoplist = list(string.punctuation) + list(pke.lang.stopwords.get('en'))
    extractor.load_document(input = text, stoplist = stoplist, language = 'en')
    extractor.candidate_selection(pos = {'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(alpha = 1.1, threshold = 0.74, method = 'average')
    keyphrases = [key for key,_ in extractor.get_n_best(n = top_n)]
    return keyphrases

@counter
def positionrank(text, top_n = 10):
    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input = text, language = 'en', normalization = None)
    extractor.candidate_selection(grammar = "NP: {<ADJ>*<NOUN|PROPN>+}", maximum_word_number = 3)
    extractor.candidate_weighting(window = 10, pos = {'NOUN', 'PROPN', 'ADJ'})
    keyphrases = [key for key,_ in extractor.get_n_best(n = top_n)]
    return keyphrases

@counter
def topicalpagerank(text, top_n = 10):
    lda_model_file = r'..\pke\models\lda-1000-semeval2010.py3.pickle.gz'
    extractor = pke.unsupervised.TopicalPageRank()
    extractor.load_document(input = text, language = 'en', normalization = None)
    extractor.candidate_selection(grammar = "NP: {<ADJ>*<NOUN|PROPN>+}")
    extractor.candidate_weighting(window = 10, pos = {'NOUN', 'PROPN', 'ADJ'}, lda_model = lda_model_file)
    keyphrases = [key for key,_ in extractor.get_n_best(n = top_n)]
    return keyphrases

@counter
def singlerank(text, top_n = 10):
    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input = text, language = 'en', normalization = None)
    extractor.candidate_selection(pos = {'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(window = 10, pos = {'NOUN', 'PROPN', 'ADJ'})
    keyphrases = [key for key,_ in extractor.get_n_best(n = top_n)]
    return keyphrases

@counter
def textrank(text, top_n = 10):
    extractor = pke.unsupervised.TextRank()
    extractor.load_document(input = text, language = 'en', normalization = None)
    extractor.candidate_weighting(window = 2, pos = {'NOUN', 'PROPN', 'ADJ'}, top_percent = 0.33)
    keyphrases = [key for key,_ in extractor.get_n_best(n = top_n)]
    return keyphrases

@counter
def topicrank(text, top_n = 10):
    extractor = pke.unsupervised.TopicRank()
    stoplist = list(string.punctuation) + list(pke.lang.stopwords.get('en'))
    extractor.load_document(input = text, stoplist = stoplist, language = 'en')
    extractor.candidate_selection(pos = {'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(threshold = 0.74, method = 'average')
    keyphrases = [key for key,_ in extractor.get_n_best(n = top_n)]
    return keyphrases


@counter
def py_textrank(nlp, text, top_n = 10):
    nlp.add_pipe('textrank')
    doc = nlp(text)
    nlp.remove_pipe('textrank')
    
    keyphrases = [
        phrase.text for phrase in doc._.phrases
    ]
    return keyphrases[:top_n]

@counter
def py_positionrank(nlp, text, top_n = 10):
    nlp.add_pipe('positionrank')
    doc = nlp(text)
    nlp.remove_pipe('positionrank')
    
    keyphrases = [
        phrase.text for phrase in doc._.phrases
    ]
    return keyphrases[:top_n]

@counter
def py_topicrank(nlp, text, top_n = 10):
    nlp.add_pipe('topicrank')
    doc = nlp(text)
    nlp.remove_pipe('topicrank')
    
    keyphrases = [
        phrase.text for phrase in doc._.phrases
    ]
    return keyphrases[:top_n]

@counter
def yake_ke(text, top_n = 10):
    custom_kw_extractor = yake.KeywordExtractor(lan = "en", n = 3, dedupLim = 0.9, dedupFunc = 'seqm', windowsSize = 1, top = 10, features=None)
    keywords = [key for key,_ in custom_kw_extractor.extract_keywords(text)]
    return keywords


def single_test():
    text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."

    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")

    print(kpminer(text))
    print(mprank(text))
    print(topicalpagerank(text))
    print(singlerank(text))
    print('\n\n')
    
    print('\n\n')
    print(textrank(text))
    print(py_textrank(nlp, text))

    print('\n\n')
    print(positionrank(text))
    print(py_positionrank(nlp, text))

    print('\n\n')
    print(topicrank(text))
    print(py_topicrank(nlp, text))
    print(yake_ke(text))
    return

def main():
    nlp = spacy.load('en_core_web_sm')
    method_name = 'textrank'
    method = { 
        'kpminer': lambda nlp, text: kpminer(text),
        'mprank': lambda nlp, text: mprank(text),
        'topicalpagerank': lambda nlp, text: topicalpagerank(text),
        'singlerank': lambda nlp, text: singlerank(text),
        'pytextrank': lambda nlp, text: py_textrank(nlp, text),
        'textrank': lambda nlp, text: textrank(text),
        'positionrank': lambda nlp, text: positionrank(text),
        'pypositionrank': lambda nlp, text: py_positionrank(nlp, text),
        'topicrank': lambda nlp, text: topicrank(text),
        'pytopicrank': lambda nlp, text: py_topicrank(nlp, text),
        'yake': lambda nlp, text: yake_ke(text)
    }

    base_path = r'..\datasets\Krapivin2009'
    input_dir = os.path.join(base_path, 'docsutf8')
    output_dir = os.path.join(base_path, f'extracted\{method_name}')
    print(os.getcwd())

    # Set the current directory to the input dir
    os.chdir(os.path.join(os.getcwd(), input_dir))

    # Get all file names and their absolute paths.
    docnames = sorted(os.listdir())
    docpaths = list(map(os.path.abspath, docnames))

    # Create the keys directory, after the names and paths are loaded.
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)

    for i, (docname, docpath) in enumerate(zip(docnames, docpaths)):

        #if i < 225: continue
        # keys shows up in docnames, erroneously.
        if docname == 'keys':
            continue
            
        print(f'Processing {i} out of {len(docnames)}...')

        # Save the output dir path
        output_dirpath = os.path.join(output_dir, docname.split('.')[0]+'.key')
        print(output_dirpath)

        with open(docpath, 'r', encoding = 'utf-8-sig', errors = 'ignore') as file, \
                open(output_dirpath, 'w', encoding = 'utf-8-sig', errors = 'ignore') as out:
            
            # Read the file and remove the newlines.
            text = file.read().replace('\n', ' ')

            # Extract the top 10 keyphrases.
            try:
                ranked_list = method[method_name](nlp, text)
                keys = '\n'.join(map(str, ranked_list) or '')
                out.write(keys)
            except Exception:
                pass

        os.system('clear')


if __name__ == '__main__': main()
