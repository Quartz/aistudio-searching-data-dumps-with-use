
import faulthandler; faulthandler.enable()

import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
import time
from tqdm import tqdm
import time
import json
from os import environ
import pandas as pd
import numpy as np
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from tensorflow.python.framework.errors_impl import InvalidArgumentError as TFInvalidArgumentError
import faiss
from annoy import AnnoyIndex


batch_size = 256
total_chunks = 37281
trees_to_build = 10

try:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        print (sess.run(c))
except (RuntimeError, TFInvalidArgumentError): 
    print("no GPU present, this'll be slow, probably")


use_module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"

g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed_module = hub.Module(use_module_url)
    embedded_text = embed_module(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

session = tf.Session(graph=g)
session.run(init_op)



def generate_embeddings (messages_in):
    if len(messages_in) == 0:
        return np.array([])
    return session.run(embedded_text, feed_dict={text_input: messages_in})


# sanity check
generate_embeddings(["My favorite kind of bagel is a toasted bagel."])


ES_INDEX_FULL_TEXT = "nycdocs"
ES_INDEX_CHUNK = "nycdocs-chunk15"
vector_dims = 512

doc_counter = 0

idx_name_chunk = {}
name_idx_chunk = {}
def vectorize_batch_chunk(lbatch, vector_index_chunk):
    global doc_counter
    
    doc_idxs = []
    for i in range(lbatch.shape[0]):
        doc_idxs.append(doc_counter)
        doc_counter += 1

    vectors = generate_embeddings(lbatch["text"])
    if len(vectors.shape) >= 2 and vectors.shape[1] > 0:
        for vec, page_num in zip(vectors, doc_idxs):
            vector_index_chunk.add_item(page_num, vec)

vector_index_chunk = AnnoyIndex(vector_dims, 'angular')
vector_index_chunk.on_disk_build(ES_INDEX_CHUNK + f"_annoy.bin")

with tqdm(total=total_chunks) as pbar:
    for j, batch in enumerate(pd.read_json('nyc_docs-sentences15.json', lines=True, chunksize=batch_size)):
        batch["smallenough"] = batch["text"].apply(lambda x: len(x) < 100000)
        batch = batch[batch["smallenough"]]
        try:
            vectorize_batch_chunk(batch, vector_index_chunk)
        except ResourceExhaustedError: 
            minibatches = np.array_split(batch, batch_size)
            for i, minibatch in enumerate(minibatches):
                try:
                    vectorize_batch_chunk(minibatch, vector_index_chunk)
                except ResourceExhaustedError:
                    continue
        pbar.update(len(batch))

vector_index_chunk.build(trees_to_build) # 10 trees
