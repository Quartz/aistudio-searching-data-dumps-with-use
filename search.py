# this model tells us _how_ we can run searches. feel free to dig in, but it's just plumbing.

import numpy as np
import unicodedata
import csv
from os.path import basename

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode("utf-8")

def index_or_error(list_, item):
    try:
        return list_.index(item)
    except ValueError:
        return "absent"

class QzUSESearch():
    def __init__(self, results, search_terms, es, es_index_full_text, seed_docs=[]):
        self.results = list(results)
        self.search_terms = search_terms
        self.clean_search_terms = [remove_accents(term).lower() for term in search_terms]
        self.seed_docs = [i.split("c")[0] for i in seed_docs]
        self.es = es
        self.es_index_full_text = es_index_full_text
        
        
    def show(self, show_seed_docs=True):
        for res, dist in self.results:
            terms_in_doc = [search_term in remove_accents(res["_source"]["text"].lower()) for search_term in self.search_terms] 
            doc_id = res["_id"].split("c")[0]
            chunk = res["_id"].split("c")[1]
            is_seed_doc = doc_id in self.seed_docs or res["_id"] in self.seed_docs
            if is_seed_doc and not show_seed_docs:
                continue
            print(res["_id"])
            print("http://example.com/{}".format(doc_id.split("c")[0]))
            print("sanity checks: ({})".format( terms_in_doc))
            print("")
    
    def sanity_check(self, targets):
        idxes = [index_or_error([chunk_res['_id'].split("c")[0] for chunk_res, dist in self.results], should_match.split("c")[0]) for should_match in targets]
        print("sanity check: these should be low-ish:" , idxes, len(self.results))
    
    def to_csv(self, csv_fn=None):
        search_term_cln = self.search_terms[0].replace(" ", "_")
        csv_fn = csv_fn if csv_fn else f"csvs/{search_term_cln}.csv"
        print(csv_fn)
        with open(csv_fn, 'w') as csvfile:
            fieldnames = [ 
                          "url", 
                          "is_seed_doc",
                          "first few words of match",
                          "chunk",
                          "distance", 
                         ] + [f"matches search term (\"{search_term}\")" for search_term in self.search_terms]
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)

            for chunk_res, dist in self.results:
                doc_id = chunk_res["_id"].split("c")[0]
                chunk_str = chunk_res["_id"].split("c")[1]
                if len(chunk_str) > 0:
                    chunk = int(chunk_str)
                else:
                    chunk = None
                
                full_text_res = self.es.get(index=self.es_index_full_text, id=doc_id)
                url = "http://example.com/{}/{}".format(doc_id.split("c")[0], full_text_res["_source"].get("routing", "") or '')
                clean_text = remove_accents(full_text_res["_source"]["text"].lower())
                terms_in_doc = [search_term in clean_text for search_term in self.clean_search_terms] 
                is_seed_doc = doc_id in self.seed_docs or chunk_res["_id"] in self.seed_docs
                
                if chunk:
                    text = chunk_res["_source"]["text"]
                else:
                    text = full_text_res["_source"]["text"][:100]
                    
                row = [
                    url,
                    is_seed_doc,
                    text,
                    chunk_str,
                    dist,
                ] + terms_in_doc
                writer.writerow(row)

class QzUSESearchFactory():
    def __init__(self, vector_index, idx_name, name_idx, es, es_index_full_text, es_index_chunk, generate_embeddings ):
        self.idx_name = idx_name
        self.name_idx = name_idx
        self.vector_index = vector_index
        self.es = es
        self.generate_embeddings = generate_embeddings
        self.es_index_full_text = es_index_full_text
        self.es_index_chunk = es_index_chunk

    def query_by_docs(self, seed_docs, search_terms=[], k=10):
        target_vectors = []
        for doc_idx in seed_docs: 
            if ":" not in doc_idx and "c" not in doc_idx: # if it's a whole doc
                res = self.es.get(index=self.es_index_full_text, id=doc_idx.split("c")[0])
                chunks = [page for j, page in enumerate(to_short_paragraphs(res["_source"]["text"]))]
                doc_avg_vec = np.mean(np.array(self.generate_embeddings(chunks)), axis=0)
                target_vectors.append(doc_avg_vec)
            elif "c" in doc_idx: # if it's a chunked doc
                chunk_vec = self.vector_index.get_item_vector(self.name_idx[doc_idx])
                target_vectors.append(chunk_vec)
            elif ":" in doc_idx:
                start, end = [int(i) for i in doc_idx.split("c")[1].split("-")]
                assert start < end
                chunks_vecs = [self.vector_index.get_item_vector(self.name_idx[doc_idx.split("c")[0] + "c" + str(i)]) for i in range(start, end + 1)]
                doc_avg_vec = np.mean(np.array(chunks_vecs), axis=0)
                target_vectors.append(doc_avg_vec)
            else:
                raise ArgumentError(f"invalid seed doc: {doc_idx}")
        avg_vec = np.average(target_vectors, axis=0)
        docs, distances = self.query_nn_with_vec(avg_vec, k)
        return QzUSESearch(zip(docs, distances), search_terms, self.es, self.es_index_full_text, seed_docs)
    
    def convert_vector(self, query): 
        query = [query]
        vector = self.generate_embeddings(query)[0]
        return vector
    
    def query_nn_with_vec(self, vector_converted, k=10):
        idxs, distances = self.vector_index.get_nns_by_vector(vector_converted, k, search_k=-1, include_distances=True)
        docs = [self.es.get(index=self.es_index_chunk, id=self.idx_name[str(doc_idx)]) for doc_idx in idxs]
        return docs, distances

    def query_nn(self, query, k=10):
        vector_converted = self.convert_vector(query)
        res = self.query_nn_with_vec(vector_converted, k)
        return res[0]
    
    def doc_avg(self, doc):
        n = 0
        chunk_vecs = []
        while 1:
            try: 
                chunk_vecs.append(self.vector_index.get_item_vector(self.name_idx[f"{doc}c{n}"]))
            except KeyError:
                break
            n += 1
        return np.mean(np.array(chunk_vecs), axis=0)

    def docs_to_avgs(self, doc_ids):
        return [n for n in [searcher.doc_avg(doc_id) for doc_id in doc_ids] if not np.isnan(n).all()]

                
    def query_by_text(self, query, k=10):
        results = self.query_nn(query, k)
        for res in results:
            doc_id = res["_id"].split("c")[0]
            url = "https://example.com/{}/{}".format(doc_id, res["_source"].get("routing", "") or '')
            full_text_res = self.es.get(index=self.es_index_full_text, id=doc_id)
            print(full_text_res["_id"])
            print(url)
            print(res["_source"]["text"])
            print("")
