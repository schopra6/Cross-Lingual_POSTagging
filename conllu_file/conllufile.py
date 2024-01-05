import os
# !pip install conllu
import conllu
import argparse

def _generate_examples(data_dir, filepaths):
    id = 0
    for path in filepaths:
        path = os.path.join(data_dir, path)
        parsed_data = []
        with open(path, "r", encoding="utf-8") as data_file:
            tokenlist = list(conllu.parse_incr(data_file))

            for sent in tokenlist:
                if "sent_id" in sent.metadata:
                    idx = sent.metadata["sent_id"]
                else:
                    idx = id

                tokens = [token["form"].replace(' ','') for token in sent]
                upos = [token["upos"] for token in sent]
                indices = [i for i, x in enumerate(upos) if x == "PUNCT" or x == "X" or x == "_"]
                # deleting tokens of junk values in corpus
                for index in sorted(indices, reverse=True):
                      del tokens[index]
                      del upos[index]
                if len(tokens) ==0:
                  # adding dummy tokens
                  tokens =['x','x'] 
                  upos =['Junk','Junk']    
                parsed_data.append(list(zip(tokens, upos)))
            return parsed_data

def write_conll_source(fstream, data):
    """
    Writes to an output stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @data a list of examples [(tokens), (labels)
    """
    for cols in data:
            fstream.write(str(cols[0])+"\t"+str(cols[1]))
            fstream.write("\n")
    fstream.write("\n")

def write_conll_target(fstream, data):
    """
    Writes to an output stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @data a list of examples [(tokens), (labels)
    """
    for cols in data:
            fstream.write(str(cols[0]))
            fstream.write("\n")
    fstream.write("\n")

def source_file(train_output_path, output):
  with open(train_output_path,'w') as file:
    for data in output:
      write_conll_source(file,data)

def target_file(target_output_path, output):
  with open(target_output_path,'w') as file:
    for data in output:
      write_conll_target(file,data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_input_path",
        type=str,
        required=True,
        help="path of input file for transformation",
    )
    parser.add_argument(
        "--train_input_file_name",
        type=str,
        required=True,
        help="input_file_name",
    )
    parser.add_argument(
        "--train_output_path",
        type=str,
        required=True,
        help="Path to save source file",
    )
    parser.add_argument(
        "--target_output_path",
        type=str,
        required=True,
        help="Path to save target file",
    )
    args = parser.parse_args()
    output = _generate_examples(args.train_input_path, [args.train_input_file_name])
    source_file(args.train_output_path, output)
    #target_file(args.target_output_path, output)