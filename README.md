# Zero Resource Cross-Lingual Part Of Speech Tagging
We investigate HMM performance on a Part Of Speech tagging trained using an artificially generated corpus of language B using language A where A is a labeled resource-rich language and B is a low-resource or unannotated language. 



	contains data in English, French and Spanish; 100k sentences
* Python version: 3.7.5

## Additional packages used: 
   * numpy, nltk.tokenize, pandas, transformers, sentencepiece, conllu

 ## Installation
This code requires: 
````
pip install -r requirements.txt 
````
Install fast_align
````
cd Easy-Label-Projection/fast_align
git clone https://github.com/clab/fast_align.git
cd fast_align
sudo apt-get install libgoogle-perftools-dev libsparsehash-dev
mkdir build
cd build
cmake ..
make
cd ../../..

````

## Dataset

The data set have been downloaded from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3687 and is available in `/ud-treebanks-v2.8`

### How-To

## Translation 

This script translates source file(English) to French,German and Spanish language to generate approximate parallel corpus.

`sh translate/translate.sh`

## Text Formatting in conllu

This script filters data from UD Treebank and forms a conll format data with only tokenized words and tags.
````cd conllu_file
sh conllufile.sh
cd ..
````

## POS Projection Method 

It enables projecting labels from source language to target language. The orignal source code had to modified to fit to our use case. The algorithm was changed to allow variety of tags to be projected than just two tags.

````commandline

cd Easy_Label_Projection
sh annotate_projection.sh 
cd ..
````

## Run VIterbi Algorithm

The viterbi algorithm learns the hidden representation of target language POS tags and associated words and then generate POS tags on train and test data.

```` sh viterbi.sh ````

## Generate evaluation

This script evaluates the performance of generated corpus from easy projection and the ones that are already annotated in UD treebank in the respective language.

## Runtime
translate.sh takes 30 mins perfile to translate having 12k sentences on GPU RTX 1080
viterbi.sh takes 10 mins for training per execution
````sh evaluation.sh ````

## References

````
@inproceedings{garcia-ferrero-etal-2022-model,
    title = "Model and Data Transfer for Cross-Lingual Sequence Labelling in Zero-Resource Settings",
    author = "Garc{\'\i}a-Ferrero, Iker  and
      Agerri, Rodrigo  and
      Rigau, German",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.478",
    pages = "6403--6416",
}
````
