export train_input_path='../ud-treebanks-v2.8/UD_English-EWT/'
export train_input_file_name='en_ewt-ud-train.conllu'
export train_output_path='../UD_Data/en_ewt-ud-train.conllu'
export target_output_path='translated_output.conllu'

python conllufile.py \
--train_input_path=$train_input_path \
--train_input_file_name=$train_input_file_name \
--train_output_path=$train_output_path \
--target_output_path=$target_output_path

export train_input_path='../ud-treebanks-v2.8/UD_Spanish-PUD/'
export train_input_file_name='es_pud-ud-test.conllu'
export train_output_path='../UD_Data/es_pud-ud-test.conllu'

python conllufile.py \
--train_input_path=$train_input_path \
--train_input_file_name=$train_input_file_name \
--train_output_path=$train_output_path \
--target_output_path=$target_output_path

export train_input_path='../ud-treebanks-v2.8/UD_French-PUD/'
export train_input_file_name='fr_pud-ud-test.conllu'
export train_output_path='../UD_Data/fr_pud-ud-test.conllu'


python conllufile.py \
--train_input_path=$train_input_path \
--train_input_file_name=$train_input_file_name \
--train_output_path=$train_output_path \
--target_output_path=$target_output_path

export train_input_path='../ud-treebanks-v2.8/UD_German-PUD/'
export train_input_file_name='de_pud-ud-test.conllu'
export train_output_path='../UD_Data/de_pud-ud-test.conllu'


python conllufile.py \
--train_input_path=$train_input_path \
--train_input_file_name=$train_input_file_name \
--train_output_path=$train_output_path \
--target_output_path=$target_output_path



export train_input_path='../ud-treebanks-v2.8/UD_German-GSD/'
export train_input_file_name='de_gsd-ud-train.conllu'
export train_output_path='../UD_Data/de_gsd-ud-train.conllu'


python conllufile.py \
--train_input_path=$train_input_path \
--train_input_file_name=$train_input_file_name \
--train_output_path=$train_output_path \
--target_output_path=$target_output_path

export train_input_path='../ud-treebanks-v2.8/UD_Spanish-GSD/'
export train_input_file_name='es_gsd-ud-train.conllu'
export train_output_path='../UD_Data/es_gsd-ud-train.conllu'


python conllufile.py \
--train_input_path=$train_input_path \
--train_input_file_name=$train_input_file_name \
--train_output_path=$train_output_path \
--target_output_path=$target_output_path

export train_input_path='../ud-treebanks-v2.8/UD_French-GSD/'
export train_input_file_name='fr_gsd-ud-train.conllu'
export train_output_path='../UD_Data/fr_gsd-ud-train.conllu'


python conllufile.py \
--train_input_path=$train_input_path \
--train_input_file_name=$train_input_file_name \
--train_output_path=$train_output_path \
--target_output_path=$target_output_path