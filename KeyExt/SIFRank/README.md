# SIFRank

This directory contains the modified code for the [SIFRank](https://github.com/sunyilgdx/SIFRank) approach.

## Modified files
The following files were modified in place, as to remove the hardcoded datasets paths,
and ensured that the approach runs in CPU mode.

* main.py
* embeddings.sent_emb_sif.py
* embeddings.word_emb_elmo

## Setup
Follow the instructions from the original repo and `pip install requirements.txt`.  
Afterwards replace the files with the modified ones.  
In `main.py`, `base_path` and `exec_path` need to be respectively set for the dataset directory and the local project path.  
In `sent_emb_sif`, `weightfile_pretrain` and `weightfile_finetune` need to be set to the respective files of the local project path.  
In `word_emb_elmo`, `options_file` and `weights_file` need to be similarly set.  
If you wish to run the `benchmark()` function you need to set the `output_path`, in `main.py` as well.  
