
python annotation_projection.py \
--source_test ../UD_Data/en_ewt-ud-train.conllu \
--target_test ../UD_Data/translation-es.txt \
--output_dir ../UD_Data/ \
--output_name es_ewt-ud-train.conllu \
--do_fastalign


python annotation_projection.py \
--source_test ../UD_Data/en_ewt-ud-train.conllu \
--target_test ../UD_Data/translation-fr.txt \
--output_dir ../UD_Data/ \
--output_name fr_ewt-ud-train.conllu \
--do_fastalign


python annotation_projection.py \
--source_test ../UD_Data/en_ewt-ud-train.conllu \
--target_test ../UD_Data/translation-de.txt \
--output_dir ../UD_Data/ \
--output_name de_ewt-ud-train.conllu \
--do_fastalign