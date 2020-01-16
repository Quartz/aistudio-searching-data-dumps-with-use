# encoding: utf-8
from tqdm import tqdm
import json
from bs4 import BeautifulSoup
from functools import reduce
from w3lib.html import remove_tags

import syntok.segmenter as segmenter

total_docs = 4251 # get this with `wc` (only used for progress bar)

total_short_paragraphs = 0
MAX_SENT_LEN = 100

def sentenceify(text):
    return [sl for l in [[''.join([t.spacing + t.value for t in s]) for s in p if len(s) < MAX_SENT_LEN] for p in segmenter.analyze(text)] for sl in l if any(map(lambda x: x.isalpha(), sl))]


def clean_html(html):
    if "<" in html and ">" in html:
        try:
            soup = BeautifulSoup(html, features="html.parser")
            plist = soup.find('plist')
            if plist:
                plist.decompose() # remove plists because ugh
            text = soup.getText()
        except:
            text = remove_tags(html)
        return '. '.join(text.split("\r\n\r\n\r\n"))
    else:
        return '. '.join(html.split("\r\n\r\n\r\n"))

# if this sentence is short, then group it with other short sentences (so you get groups of continuous short sentences, broken up by one-element groups of longer sentences)
def short_sentence_grouper_bean_factory(target_sentence_length): # in chars
    def group_short_sentences(list_of_lists_of_sentences, next_sentence):
        if not list_of_lists_of_sentences:
            return [[next_sentence]]
        if len(next_sentence) < target_sentence_length:
           list_of_lists_of_sentences[-1].append(next_sentence)
        else:
            list_of_lists_of_sentences.append([next_sentence])
            list_of_lists_of_sentences.append([])
        return list_of_lists_of_sentences
    return group_short_sentences


def overlap(document_tokens, target_length):
    """ pseudo-paginate a document by creating lists of tokens of length `target-length` that overlap at 50%

    return a list of `target_length`-length lists of tokens, overlapping by 50% representing all the tokens in the document 
    """

    overlapped = []
    cursor = 0
    while len(' '.join(document_tokens[cursor:]).split()) >= target_length:
      overlapped.append(document_tokens[cursor:cursor+target_length])
      cursor += target_length // 2
    return overlapped


def sentences_to_short_paragraphs(group_of_sentences, target_length, min_shingle_length=10):
    """ outputting overlapping groups of shorter sentences 
    
        group_of_sentences = list of strings, where each string is a sentences
        target_length = max length IN WORDS of output sentennces
        min_shingle_length = don't have sentences that differ just in the inclusion of a sentence of this size
    """
    if len(group_of_sentences) == 1:
        return [' '.join(group_of_sentences[0].split())]
    sentences_as_words = [sent.split() for sent in group_of_sentences]
    sentences_as_words = [sentence for sentence in sentences_as_words if [len(word) for word in sentence].count(1) < (len(sentence) * 0.5) ]
    paragraphs = []
    for i, sentence in enumerate(sentences_as_words[:-1]):
        if i > 0 and len(sentence) < min_shingle_length  and len(sentences_as_words[i-1]) < min_shingle_length and i % 2 == 0:
            continue # skip really short sentences if the previous one is also really short (but not so often that we lose anything )
        buff = list(sentence) # just making a copy.
        for subsequent_sentence in sentences_as_words[i+1:]:
            if len(buff) + len(subsequent_sentence) <= target_length:
                buff += subsequent_sentence
            else:
                break
        paragraphs.append(buff)
    return [' '.join(graf) for graf in paragraphs]


def to_short_paragraphs(text, paragraph_len=15, min_sentence_len=8): # paragraph_len in words, min_sentence_len in chars
    sentences = sentenceify( clean_html(text) )
    grouped_sentences = reduce(short_sentence_grouper_bean_factory(150) , sentences, [])
    return [sl for l in [sentences_to_short_paragraphs(group, paragraph_len) for group in grouped_sentences if len(group) >= 2 or (len(group) > 0 and len(group[0]) > min_sentence_len)] for sl in l if sl]

paragraph_target_length = 15

if __name__ == "__main__":
    with open(f"nyc_docs-sentences{paragraph_target_length}.json", 'w') as writer: 
        with open('nyc_docs.jsonl', 'r') as reader:
            for i, line_json in tqdm(enumerate(reader), total=total_docs):
                line = json.loads(line_json)
                text = line["_source"]["content"][:1000000]
                for j, page in enumerate(to_short_paragraphs(text, paragraph_target_length)):
                    total_short_paragraphs += 1
                    writer.write(json.dumps({
                        "text": page, 
                        "_id": line["_id"], 
                        "chonk": j,
                        # "routing": line.get("_routing", None),
                        # "path": line["_source"]["path"]
                        }) + "\n")
    print(f"total paragraphs: {total_short_paragraphs}")
