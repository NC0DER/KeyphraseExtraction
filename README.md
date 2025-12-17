# Keyword & Keyphrase Extraction Review

This repository hosts code for the papers:
* [A literature review of keyword and keyphrase  extraction -]() - [Download]()
* [A comparative assessment of state-of-the-art methods for multilingual unsupervised keyphrase extraction](https://link.springer.com/chapter/10.1007/978-3-030-79150-6_50) - [Download](https://github.com/NC0DER/KeyphraseExtraction/releases/tag/KeyphraseExtractionv1.0)  

## Datasets
Available in [this link](https://drive.google.com/drive/folders/1ziElrM1Y3Wp1vLK21OPtsN7Da-bbR7Sb)

## Disclaimer 
This repository contains code for the evaluated approaches.
The code for these approaches belongs to their respective authors.
Some code files were modified to enable the evaluation.
These modifications include:
* Removing hardcoded paths.
* Setting `cpu-only` mode for approaches that require a lot of `GPU VRAM`.
* Updating code to run from `Python 2` to `Python 3`.
* Amend errors related to old packages or functions with wrong parameters.
* Disabling stemming performed early by certain approaches in their keyphrase extraction step, 
  as to use a common stemmer later in the evaluation process.

## Test Results
Configure `KeyExt\config.py` and run `KeyExt.py`.

## Installation
* `Python 3` (min. version 3.7), `pip3` (& `py` launcher Windows-only).
* Follow the install instructions in each subdirectory.

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
