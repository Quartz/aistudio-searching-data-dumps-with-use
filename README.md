# Searching Bill de Blasio's Emails with the Universal Sentence Encoder

By Jeremy B. Merrill, [Quartz](https://www.qz.com)

As part of Quartz's participation in the Luanda Leaks investigation, we built an system for searching large heterogenous document sets with AI. For more info, LINK TK.

Here's the code demo so you can reproduce our work. For this demo, we'll be searching in a set of a few thousand emails between Bill de Blasio and some advisors, informally called the "Agent of the City documents."

This isn't a full library or off-the-shelf software, but rather a demonstration that you can adapt for yourself.

## how to
There are a few steps you'll have to take to run the demo.

1. Get a computer with Python 3.7. You probably want a computer with a GPU, but it's not strictly necessary. Without a GPU, this'll be very slow.
2. Split the provided `nyc_docs.jsonl` file (which has a text copy of every document in the document set) into sentence-size chunks by running `python to_sentences.py`. We need to split the documents into sentence-length chunks because Universal Sentence Encoder can only handle blocks of text shorter than 128 words.
3. Index the chunks and the full documents to ElasticSearch by running `python to_es.py`.
4. Embed the chunks into vector space with USE and index those with Annoy by running `python to_annoy.py`.
5. Run the "Searching with USE.ipynb" Jupyter notebook to run your own searches.


--
## about tech choices

*Universal Sentence Encoder* is a tensorflow library that, without any additional training, embeds sentences in several languages into a shared vector representation. Basically, sentences with a similar meaning have vectors that are close together.

*Annoy* is a Python library that makes nearest neighbor searches for vectors very fast. This lets us index thousands, or millions, of vectors and instantly get back which ones are close by.

Combined, these tools let us ask the computer, "what sentences mean something similar to this?" and we'll get back high quality results, even if the similar sentences don't overlap in any keywords.