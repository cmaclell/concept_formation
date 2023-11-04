import os
import re
import json
from collections import Counter
from tqdm import tqdm

# import sentencepiece as spm
import spacy

# sp = spm.SentencePieceProcessor(model_file='spiece.model')
nlp = spacy.load("en_core_web_sm", disable = ['parser'])
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')

def preprocess_text(text, test=False):
    if test:
        punc = re.compile(r"[^_a-zA-Z,.!?:;\s]")
    else:
        punc = re.compile(r"[^a-zA-Z,.!?:;\s]")
    whitespace = re.compile(r"\s+")
    text = punc.sub("", text)
    text = whitespace.sub(" ", text)
    text = text.strip().lower()
    return text

# def process_text(text, test=False):
#     # out = preprocess_text(text, text)
#     # out = nlp(out)
#     # out = [token.lemma_ for token in out if (not token.is_punct
#     #                                          and not token.is_stop)]
#     # return out
#     punc = re.compile(r"[^a-zA-Z,.!?:;\s]")
#     text = punc.sub("", text)
#     out = sp.encode(text.lower(), out_type=str)
#     return out

def process_text(text):
    text = preprocess_text(text) 
    text = nlp(text)
    text = [token.lemma_.lower() for token in text if (not token.is_punct and
                                                       not token.is_stop)]
    # story = [token.text.lower() for token in story if (not token.is_punct)]
    return text


def daily_texts(limit=None):

    training_dir = "/Users/cmaclellan3/Projects/concept_formation/concept_formation/tests/cnn-daily/daily-stories"

    for path, subdirs, files in os.walk(training_dir):

        files = [f for f in files if re.search(r'^[a-zA-Z0-9]*.story$', f)]

        if limit is None:
            limit = len(files)

        for idx, name in enumerate(tqdm(files[:limit])):
            with open(os.path.join(path, name), 'r', encoding='latin-1') as fin:
                text = ""
                for line in fin:
                    text += line

                output = process_text(text)
                # print(output)

                yield output

def cnn_texts(limit=None):

    training_dir = "/Users/cmaclellan3/Projects/concept_formation/concept_formation/tests/cnn-daily/cnn-stories"

    for path, subdirs, files in os.walk(training_dir):

        files = [f for f in files if re.search(r'^[a-zA-Z0-9]*.story$', f)]

        if limit is None:
            limit = len(files)

        for idx, name in enumerate(tqdm(files[:limit])):
            with open(os.path.join(path, name), 'r', encoding='latin-1') as fin:
                text = ""
                for line in fin:
                    text += line

                output = process_text(text)
                # print(output)

                yield output

def training_texts(limit=None):

    training_dir = "/Users/cmaclellan3/Projects/Microsoft-Sentence-Completion-Challenge/data/raw_data/Holmes_Training_Data"

    for path, subdirs, files in os.walk(training_dir):

        if limit is None:
            limit = len(files)

        for idx, name in enumerate(files[:limit]):
            print("Processing file {} of {}".format(idx, len(files)))
            if not re.search(r'^[A-Z0-9]*.TXT$', name):
                continue
            print(name)
            with open(os.path.join(path, name), 'r', encoding='latin-1') as fin:
                skip = True
                text = ""
                for line in fin:
                    if not skip and not "project gutenberg" in line.lower():
                        text += line
                    elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
                        skip = False

                output = process_text(text)
                # print(output)

                yield output

def load_holmes_data(limit=None):

    if not os.path.isfile("holmes_stories.json"):
        print("Reading and preprocessing holmes stories.")
        stories = list(training_texts(limit=limit))
        with open("holmes_stories.json", "w") as fout:
            json.dump(stories, fout, indent=4)
        print("done.")
    else:
        print("Loading preprocessed holmes stories.")
        with open("holmes_stories.json", "r") as fin:
            stories = json.load(fin)
        print("done.")

    return stories

def load_cnn_daily_data():
    if not os.path.isfile("cnn_daily_stories.json"):
        print("Reading and preprocessing cnn/daily stories.")
        stories = list(cnn_texts()) 
        # stories += list(daily_texts())
        with open("cnn_daily_stories.json", "w") as fout:
            json.dump(stories, fout, indent=4)
        print("done.")
    else:
        print("Loading preprocessed cnn/daily stories.")
        with open("cnn_daily_stories.json", "r") as fin:
            stories = json.load(fin)
        print("done.")

    return stories


if __name__ == "__main__":

    stories = load_holmes_data()

    overall_freq = Counter([w for s in stories for w in s])

    print()
    print("MOST COMMON")
    print(overall_freq.most_common(100))

    #print()
    # print("LEAST COMMON")
    # print(overall_freq.most_common()[10000:])

    for s in stories:
        print()
        print(s[:200])
