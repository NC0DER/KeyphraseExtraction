import os
import launch
import pathlib

base_path = '../datasets/DUC-2001/'
input_dir = os.path.join(base_path, 'docsutf8')
output_dir = os.path.join(base_path, 'extracted/embedrank')

embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()

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
            kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, text, 10, 'en')
            keys = "\n".join(kp1[0] or '')
            out.write(keys)
        except:
            pass

    os.system('clear')