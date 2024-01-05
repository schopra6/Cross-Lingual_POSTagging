export INPUT_PATH='/ud-treebanks-v2.8/UD_English-EWT/en_ewt-ud-train.conllu'
export OUTPUT_PATH='/UD_Data/translation-es.txt'
export MODEL='Helsinki-NLP/opus-mt-en-es'

python translate.py \
--input_path=$INPUT_PATH \
--output_path=$OUTPUT_PATH \
--model=$MODEL


export OUTPUT_PATH='/UD_Data/translation-fr.txt'
export MODEL='Helsinki-NLP/opus-mt-en-fr'

python translate.py \
--input_path=$INPUT_PATH \
--output_path=$OUTPUT_PATH \
--model=$MODEL


export OUTPUT_PATH='/UD_Data/translation-de.txt'
export MODEL='Helsinki-NLP/opus-mt-en-de'

python translate.py \
--input_path=$INPUT_PATH \
--output_path=$OUTPUT_PATH \
--model=$MODEL
