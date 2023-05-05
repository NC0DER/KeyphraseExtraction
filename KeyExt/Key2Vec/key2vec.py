import os
import pathlib
import key2vec

def main():
    glove = key2vec.glove.Glove('./data/glove.6B.50d.txt')
    base_path = '../datasets/DUC-2001'
    input_dir = os.path.join(base_path, 'docsutf8')
    output_dir = os.path.join(base_path, 'extracted/key2vec')

    # Set the current directory to the input dir
    os.chdir(os.path.join(os.getcwd(), input_dir))

    # Get all file names and their absolute paths.
    docnames = sorted(os.listdir())
    docpaths = list(map(os.path.abspath, docnames))

    # Create the keys directory, after the names and paths are loaded.
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)

    for i, (docname, docpath) in enumerate(zip(docnames, docpaths)):

        if i < 292: continue
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
                m = key2vec.key2vec.Key2Vec(text, glove)
                m.extract_candidates()
                m.set_theme_weights()
                m.build_candidate_graph()
                ranked_list = m.page_rank_candidates(top_n = 10)

                keys = "\n".join(map(str, ranked_list) or '')
                out.write(keys)
            except:
                pass

        os.system('clear')


if __name__ == "__main__": main()
