
import faulthandler; faulthandler.enable()

import time
from tqdm import tqdm
import time
import json
from os import environ
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search

ES_INDEX_FULL_TEXT = "nycdocs-use"
FIRST = False
ES_INDEX_CHUNK = "nycdocs-use-chunk128"
vector_dims = 512
batch_size = 512
total_chunks = 37281 # get this with `wc nyc_docs_paragraphs.json`
total_docs = 4251


es = Elasticsearch()

doc_counter = 0

idx_name_chunk = {}
name_idx_chunk = {}

def es_index_batch_chunk(lbatch):
    global doc_counter

    records = []
    for body in lbatch.to_dict(orient='records'):
        id_ = body["_id"] + "c" + str(body["chonk"])
        idx_name_chunk[doc_counter] = id_
        name_idx_chunk[id_] = doc_counter
        body["page"] = doc_counter
        body["_index"] = ES_INDEX_CHUNK
        del body["smallenough"]
        body["doc_id"] = body["_id"]
        body["_id"] = id_
        records.append(body)
        doc_counter += 1
    res = helpers.bulk(es, records, chunk_size=len(records), request_timeout=200)


import pandas as pd
import numpy as np
with tqdm(total=total_chunks) as pbar:
    for j, batch in enumerate(pd.read_json('nyc_docs-sentences15.json', lines=True, chunksize=batch_size)):
        batch["smallenough"] = batch["text"].apply(lambda x: len(x) < 100000)
        batch = batch[batch["smallenough"]]
        es_index_batch_chunk(batch)
        pbar.update(len(batch))

with open(ES_INDEX_CHUNK + "_idx_name.json", 'w') as f:
    f.write(json.dumps(idx_name_chunk))
with open(ES_INDEX_CHUNK + "_name_idx.json", 'w') as f:
    f.write(json.dumps(name_idx_chunk))

# also put the full documents into ES
with open('nyc_docs.jsonl', 'r') as reader:
    for i, line_json in tqdm(enumerate(reader), total=total_docs):
        line = json.loads(line_json)
        body = {
            "text": line["_source"]["content"][:1000000], 
            "routing": line.get("_routing", None),
            }
        es.index(index=ES_INDEX_FULL_TEXT, id=line["_id"], body=body)



