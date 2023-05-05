import os
import pathlib
import operator
import string
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
import numpy as np
import subprocess


_STOP_WORDS = [
'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 
'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although',
'always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another',
'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',
'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been', 
'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 
'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 
'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 
'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 
'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 
'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 
'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 
'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 
'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 
'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'iff', 'in', 'inc', 
'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 
'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 
'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 
'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 
'nevertheless', 'next', 'nine', 'no', 'non', 'nobody', 'none', 'noone', 'nor', 'not', 
'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only',
'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out',
'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same',
'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 
'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 
'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 
'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 
'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 
'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 
'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 
'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 
'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 
'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 
'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
'yourselves', 'the', 'zj', 'zi', 'yj', 'yi', 'xi', 'xj', 'xixj', 'xjxi', 'yiyj', 
'yjyi', 'zizj', 'zjzi']

stop = set(stopwords.words('english'))

_WORD_MIN_LENGTH = 3
_WORD_MAX_LENGTH = 35
_NUM_ITERATIONS = 50
_DIM_VECTOR = 50
_FREQUENCY = 1


def generate(vocab_file, vectors_file):

    print(os.getcwd())
    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def extract_keyphrases(docpath, docname, top_n=10):
    #if not docname.endswith('txt'):
        #return []

    # Modify the docpath as to not include the docname.
    docpath = docpath.rsplit('/', 1)[0]+'/'

    # Word vectors production.
    subprocess.call(["../RVA/glove/demo.sh", docpath+docname, docname, str(_DIM_VECTOR), str(_NUM_ITERATIONS)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    # Word vectors loading.
    W, vocab, ivocab = generate("vocab.txt"+docname+str(_DIM_VECTOR)+str(_NUM_ITERATIONS), "vectors"+docname+str(_DIM_VECTOR)+str(_NUM_ITERATIONS)+'.txt')
    
    # We use as an abstract, the first 300 words of the original file.
    # Because most of the datasets used don't have a dedicated abstract section.
    with open(docpath+docname, 'r', encoding='utf-8-sig', errors='ignore') as myfile:
        abstract_text = myfile.read()[:300]
        lowers = abstract_text.lower()
        no_punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
        temp = no_punctuation.split(' ')
        tempn = nltk.word_tokenize(no_punctuation)
      
    doc_unigrams = []
    tokens = []
    bbigrams = []
    ttrigrams = []
    no_punctuation_unstemmed = ''
         
    for x in tempn:
        no_punctuation_unstemmed += x + ' '
        tokens.append(x)
    for token in tokens:
        token = token.strip().lower()
        token = token.strip(string.digits)
        if len(token) >= _WORD_MIN_LENGTH and len(token) <= _WORD_MAX_LENGTH and '!' not in token and '@' not in token and '#' not in token and '$' not in token and '*' not in token and '=' not in token and '+' not in token and '\\x' not in token and '.' not in token and ',' not in token and '?' not in token and '>' not in token and '<' not in token and '&' not in token and not token.isdigit() and token not in _STOP_WORDS and token not in stop and '(' not in token and ')' not in token and '[' not in token and ']' not in token and '{' not in token and '}' not in token and '|' not in token and token not in doc_unigrams:                                                   
            doc_unigrams.append(token)

    n = 2
    bigrams = ngrams(tokens, n)
    for bi in bigrams:
        token1 = bi[0].lower()
        token2 = bi[1].lower()
        if token1 in doc_unigrams and token2 in doc_unigrams and not (len(token1)<=3 and len(token2)<=3):
            big = token1+' '+token2
            bitu = (token1, token2)
            if no_punctuation_unstemmed.count(big.strip())>=(_FREQUENCY):
                bbigrams.append(bitu)

    n = 3
    trigrams = ngrams(tokens, n)
    for tri in trigrams:
        token1 = tri[0].lower()
        token2 = tri[1].lower()
        token3 = tri[2].lower()
        if token1 in doc_unigrams and token2 in doc_unigrams and token3 in doc_unigrams and not (len(token1)<=3 and len(token2)<=3 and len(token3)<=3):
            big = token1+' '+token2+' '+token3
            tritu = (token1, token2, token3)
            if no_punctuation_unstemmed.count(big.strip())>=(_FREQUENCY):
                ttrigrams.append(tritu)

    
    _NUM_KEYWORDS = int(len(set(temp))/3)
    count_words = 0
    final_doc_vector = np.zeros((_DIM_VECTOR))
    word_vector = np.zeros((_DIM_VECTOR))
    for word in tempn:
        if word in vocab and word in doc_unigrams:
            word_vector = W[vocab[word], :]
            final_doc_vector += word_vector
            count_words += 1 
          
            
    final_doc_vector /= count_words 

    #Calculation of cosine similarity between mean_vec and every word - Scoring unigrams.
    dict_cand_sim = {}
    for word in tempn:
        if word in vocab and word in doc_unigrams:
            word_vector = W[vocab[word], :]  
            dict_cand_sim[str(word)] = np.dot(final_doc_vector, word_vector)/(np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector))

                  
    #Scoring bigrams and trigrams.
    for tri in ttrigrams:
        token1 = tri[0]
        token2 = tri[1]
        token3 = tri[2]
        score = 0.0
        if token1 in dict_cand_sim.keys():
            score += dict_cand_sim[token1]
        if token2 in dict_cand_sim.keys():
            score += dict_cand_sim[token2]
        if token3 in dict_cand_sim.keys():
            score += dict_cand_sim[token3]  
        dict_cand_sim[str(token1+' '+token2+' '+token3)] = score   
            
    for bi in bbigrams:
        token1 = bi[0]
        token2 = bi[1]
        score = 0.0
        if token1 in dict_cand_sim.keys():
            score += dict_cand_sim[token1]
        if token2 in dict_cand_sim.keys():
            score += dict_cand_sim[token2]
        dict_cand_sim[str(token1+' '+token2)] = score
           
    sorted_x = sorted(dict_cand_sim.items(), key=operator.itemgetter(1))
    sorted_x = sorted_x[-_NUM_KEYWORDS:]
    
    return sorted_x[:top_n]


def main():

    base_path = '../datasets/DUC-2001'
    input_dir = os.path.join(base_path, 'docsutf8')
    output_dir = os.path.join(base_path, 'extracted/rva')


    # Set the current directory to the input dir
    os.chdir(os.path.join(os.getcwd(), input_dir))

    # Get all file names and their absolute paths.
    docnames = sorted(os.listdir())
    docpaths = list(map(os.path.abspath, docnames))

    # Create the keys directory, after the names and paths are loaded.
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)


    for i, (docname, docpath) in enumerate(zip(docnames, docpaths)):

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
                ranked_list = [key for key,_ in extract_keyphrases(docpath, docname, top_n=10)]
                keys = "\n".join(map(str, ranked_list) or '')
                out.write(keys)
            except:
                pass

            

        os.system('clear')

        
if __name__ == '__main__': main()
