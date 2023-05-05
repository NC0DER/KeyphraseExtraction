# Classical Approaches

This directory contains classical unsupervised approaches, which do not utilize word embeddings.  
These include `YAKE!`, `KPMiner`, `MPRank`, `PositionRank`, `TopicalPageRank`, `SingleRank`, `TextRank` and `TopicRank`.  

## Setup
In order to run this script you need to:
```
pip install pke
pip install pytextrank
pip install spacy
pip install git+https://github.com/LIAAD/yake
```
The `en_core_web_sm` model for the respective `spacy` version needs to be installed, since it is used by [pytextrank](https://github.com/DerwenAI/pytextrank).  
`TopicalPagerank` and `KPMiner` use a `lda_model_file` and a `weights_file`respectively, which can be obtained from the [pke](https://github.com/boudinfl/pke) repo.  
After they are obtained, their respective paths and the `base_path` for the dataset directory should be set in `main.py`.  
