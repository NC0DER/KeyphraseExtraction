#from __future__ import division
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import PositionRank
from gensim.models import KeyedVectors
import evaluation
import process_data
import os
from os.path import isfile, join
import pathlib
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import pickle

def ensure_dir(dirName):
    if not os.path.exists(dirName):
        print('making dir: ' + dirName)
        os.makedirs(dirName)

def load_obj(filePath):
    with open(filePath, 'rb') as f:
        return pickle.load(f)

def main():
    # Initialize parameters.
    topK = 10
    window = 10
    phrase_type = 'ngrams'
    emb_dim = 768
    theme_mode = 'adj_noun_title'
    model_name = 'scibert'

    # Initialize paths.
    dataset_dir = r'..\datasets\Krapivin2009'
    input_dir = os.path.join(dataset_dir, 'docsutf8')
    output_dir = os.path.join(dataset_dir, 'extracted\kprank')
    emb_dir = os.path.join(dataset_dir, f'{model_name}_emb_fulltext_title')
        
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

        #if i < 115: continue

        print(f'Processing {i} out of {len(docnames)}...')
        
        # Form the output path.
        output_path = os.path.join(output_dir, docname.split('.')[0]+'.key')
        print(output_path)

        # Process the data of the document.
        text = process_data.read_input_file(docpath)

        # Load the embeddings.
        emb_path = os.path.join(emb_dir, f'{docname}_fulltext.pkl')
        embeddings = load_obj(emb_path)
        model = PositionRank.PositionRank(text, window, phrase_type, emb_dim, embeddings)

        # Run the model.
        model.get_doc_words()
        model.candidate_selection()
        model.candidate_scoring(theme_mode = theme_mode, update_scoring_method = False)
        keyphrases = model.get_best_k(topK)[:10]
        
        # Write the keyphrases to a file.
        keys = '\n'.join(map(str, keyphrases) or '')
        with open(output_path, 'w', encoding = 'utf-8-sig', errors = 'ignore') as out:
            out.write(keys)

        os.system('clear')
    return


if __name__ == "__main__": main()