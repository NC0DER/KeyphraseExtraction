import os
import numpy as np
import pandas as pd
import KeyExt.models
import KeyExt.metrics
import KeyExt.utils

def run_experiments(dirpath, outpath, top_n):
    # Initialize the spacy and keybert models.
    nlp_model, bert_model = KeyExt.utils.load_models()

    # Make a list of all subdirectories.
    directories = next(os.walk(dirpath))[1][0:]
    data = []

    for i, directory in enumerate(directories):
        print(f'Processing {i} in {len(directories)} datasets.')
        # Initialize the empty document set.
        documents = {}
        language = ''

        # Change current working directory to the dataset directory.
        os.chdir(os.path.join(dirpath, directory))

        # Find the language of the current dataset.
        with open(os.path.join(os.getcwd(), 'language.txt'), 
                  'r', encoding = 'utf-8-sig', errors = 'ignore') as text:
            language = text.read().rstrip().lower()
        
        # Find document / key filenames and paths.
        # Each time we enter the subdirectory 
        # of the current working directory.
        # First the docs then the keys subdirectory.

        os.chdir(os.path.join(os.getcwd(), 'docsutf8'))
        docnames = sorted(os.listdir())
        docpaths = list(map(os.path.abspath, docnames))

        os.chdir(os.path.join(os.getcwd(), '../keys'))
        keynames = sorted(os.listdir())
        keypaths = list(map(os.path.abspath, keynames))

        for (docname, docpath, keypath) in zip(docnames, docpaths, keypaths):
            with open(docpath, 'r', encoding = 'utf-8-sig', errors = 'ignore') as text, \
                 open(keypath, 'r', encoding = 'utf-8-sig', errors = 'ignore') as keys:
                documents[docname] = (
                    [text.read().replace('\n', ' ')]
                    + [keys.read().splitlines()]
                )

        mean_f1_measures = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Each method produces tags has a default n-gram range from 1 to 3.
        total = len(documents)
        skipped = 0
        for j, (key, (text, actual_tags)) in enumerate(documents.items()):
            # Keybert needs the text to be larger in terms than 2 * top_n.
            if len(text.split()) < 2 * top_n:
                skipped += 1
                continue

            print(f'Processing {j} in {total} documents.')
            for k, predicted_tags in enumerate([
                KeyExt.models.tfidfvectorizer(text, top_n = top_n),
                KeyExt.models.rake(text, top_n = top_n),
                KeyExt.models.yake(text, dedupFunc = 'seqm', windowsSize = 1, top_n = top_n),
                KeyExt.models.keybert(text, bert_model, measure = 'mmr', diversity = 0.7, top_n = top_n),
                KeyExt.models.keybert(text, bert_model, measure = 'maxsum', diversity = 0.7, top_n = top_n),
                KeyExt.models.textrank(text, nlp_model, top_n = top_n), 
                KeyExt.models.singlerank(text, top_n = top_n)
            ]):
                # Convert the tags to lowercase, strip punctuation and then apply stemming.
                actual_tags = KeyExt.utils.preprocess(actual_tags, language)
                predicted_tags = KeyExt.utils.preprocess(predicted_tags, language)

                # All of f1 measures from each method and text are being summed,
                # separately for each method, the mean is calculated in the list 
                # comprehension, found below this nested loop.
                mean_f1_measures[k] += KeyExt.metrics.f1_measure_k (
                    actual_tags, predicted_tags, k = top_n, partial = True
                )
            KeyExt.utils.clear_screen()

        mean_f1_measures = [
            sum_f1_measure / (total - skipped) 
            for sum_f1_measure in mean_f1_measures
        ]
        # Append f1 measure list for each document to the data list of lists,
        # each list has the dataset name prepended at the start of the row.
        data.append([directory] + mean_f1_measures)
        KeyExt.utils.clear_screen()
        break # Debug line

    # Construct the dataframe.
    df = pd.DataFrame(data, columns = [
        f'pF1@{top_n}','Tfidfvectorizer', 'Rake', 'Yake-seqm',
        'Keybert-mmr', 'Keybert-maxsum',
        'Textrank', 'Singlerank'
    ])
    # Set the index to the first column and save to excel.
    df.set_index([f'pF1@{top_n}']).to_excel(outpath)
