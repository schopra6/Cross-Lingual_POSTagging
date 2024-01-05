import re
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import torch 
import conllu
def translator(mod):
    """
    returning translator object 
    """

    model = AutoModelForSeq2SeqLM.from_pretrained(mod)
    #device = torch.device('cuda:0')
    #model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(mod)
    translator = pipeline("translation", model=model, tokenizer=tokenizer,device=0)
    return translator

def tf_lower_and_split_punct(text):
  text = text.lower()
  text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
  return text


def convert_txt_to_array(input_path):
    """
    spliting sentence into tokens
    """

    with open(input_path) as file:
        text = file.read()
    sentences = text.split("\n")
    for i in range(len(sentences)):
        sentences[i] = tf_lower_and_split_punct(sentences[i])
    #print(sentences)
    return sentences

def generate_examples( filepath):
        """
        Args:
          filepath: path where data is
        Returns:
           list of sentences from the corpus  
        """

        parsed_data = []
        with open(filepath, "r", encoding="utf-8") as data_file:
            tokenlist = list(conllu.parse_incr(data_file))
            
            for sent in tokenlist:
                tokens = [token["form"] for token in sent]
                if "text" in sent.metadata:
                     txt = sent.metadata["text"]
                else:
                     txt = " ".join(tokens)
                parsed_data.append(txt)
            return parsed_data



def translation_file(sentences, output_path, translator):
    with open(output_path, "w") as f:
        for i in tqdm(sentences):
            ans = translator(i)
            f.write(ans[0]["translation_text"] + "\n")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path of input sentences for translation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save translated sentence",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="translation language e.g = en(input) -> fr(output)"
             "model: = Helsinki-NLP/opus-mt-en-fr"
    )
    args = parser.parse_args()
    sentences = generate_examples(args.input_path)
    trans = translator(args.model)
    translation_file(sentences, args.output_path, trans)
