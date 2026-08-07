from tqdm import tqdm
import ir_datasets
import json
from datasets import DownloadManager
from collections import defaultdict
import gzip
import pickle
from pathlib import Path
import requests
import sys
import random
from datasets import load_dataset


IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"


def read_collection(collection_path: str, text_fields=["text"], classification_labels: bool=False):
    doc_dict = {}
    doc_label_dict = {}
    if collection_path.startswith(IRDS_PREFIX):
        irds_name = collection_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for doc in tqdm(
            dataset.docs_iter(),
            desc=f"Loading doc collection from ir_datasets: {irds_name}",
        ):
            doc_id = doc.doc_id
            texts = [getattr(doc, field) for field in text_fields]
            text = " ".join(texts)
            doc_dict[doc_id] = text
    elif collection_path.startswith(HFG_PREFIX):
        hfg_name = collection_path.replace(HFG_PREFIX, "")
        dataset = load_dataset(hfg_name)
        for row in tqdm(
            dataset["passage"],
            desc=f"Loading data from HuggingFace datasets: {hfg_name}",
        ):
            doc_dict[row["id"]] = row["text"]
    else:
        collection_paths = collection_path.split(',')
        for idx, collection_path in enumerate(collection_paths):
            if ':' in collection_path:
                collection_path, sample_ratio = collection_path.split(':')
                sample_ratio = float(sample_ratio)
            else:
                sample_ratio = 1
            header = f"DS_{idx}_"
            with open(collection_path, "r") as f:
                for line in tqdm(f, desc=f"Reading doc collection from {collection_path}"):
                    parts = line.strip().split("\t")
                    doc_id, doc_text = parts[:2]
                    doc_dict[header+doc_id] = doc_text
                    if classification_labels:
                        doc_label_dict[header+doc_id] = parts[2:]
    return doc_dict if not classification_labels else doc_dict, doc_label_dict


def read_queries(queries_path: str, text_fields=["text"], classification_labels: bool=False):
    queries = []
    labels = []
    if queries_path.startswith(IRDS_PREFIX):
        irds_name = queries_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for query in tqdm(
            dataset.queries_iter(),
            desc=f"Loading queries from ir_datasets: {queries_path}",
        ):
            query_id = query.query_id
            texts = [getattr(query, field) for field in text_fields]
            text = " ".join(texts)
            queries.append((query_id, text))
    else:
        queries_pathes = queries_path.split(',')
        for idx, queries_path in enumerate(queries_pathes):
            header = f"DS_{idx}_"
            with open(queries_path, "r") as f:
                for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                    parts = line.strip().split("\t")
                    query_id, query_text = parts[:2]
                    queries.append((header+query_id, query_text))
                    if classification_labels:
                        labels.append((header+query_id, parts[2:]))
    return queries if not classification_labels else queries, labels


def read_qrels(qrels_path: str, rel_threshold=0):
    qid2pos = {}
    if qrels_path.startswith(IRDS_PREFIX):
        irds_name = qrels_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for qrel in dataset.qrels_iter():
            if qrel.relevance > rel_threshold:
                qid, did = qrel.query_id, qrel.doc_id
                if not qid in qid2pos:
                    qid2pos[qid] = []
                qid2pos[qid].append(did)
    else:
        qrels = json.load(open(qrels_path, "r"))
        for qid in qrels:
            qid2pos[str(qid)] = [str(did) for did in qrels[qid]]
    return qid2pos


file_map = {
    "sentence-transformers/msmarco-hard-negatives": "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
}


def read_ce_score(ce_path: str):
    if ce_path.startswith(HFG_PREFIX):
        hf_name = ce_path.replace(HFG_PREFIX, "")
        _url = file_map[hf_name]
        dl_manager = DownloadManager()
        ce_path = dl_manager.download(_url)
    res = {}
    with gzip.open(ce_path, "rb") as f:
        data = pickle.load(f)
        for qid in tqdm(data, desc=f"Preprocessing data from {ce_path}"):
            res[str(qid)] = {str(did): data[qid][did] for did in data[qid]}
    return res


def read_triplets(triplet_path: str):
    triplets = []
    query2pos = defaultdict(set)
    query2neg = defaultdict(set)
    if triplet_path.startswith(IRDS_PREFIX):
        irds_name = triplet_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for docpair in tqdm(
            dataset.docpairs_iter(), f"Loading triplets from ir_datasets: {irds_name}"
        ):
            qid, pos_id, neg_id = docpair.query_id, docpair.doc_id_a, docpair.doc_id_b
            triplets.append((qid, pos_id, neg_id))
            query2pos[qid].add(pos_id)
            query2neg[qid].add(neg_id)
    else:
        triplet_paths = triplet_path.split(',')
        for idx, triplet_path in enumerate(triplet_paths):
            header = f"DS_{idx}_"
            _query2pos = defaultdict(set)
            _query2neg = defaultdict(set)
            ratio = 1
            if ':' in triplet_path:
                triplet_path, ratio = triplet_path.split(':')
                ratio = float(ratio)
                assert ratio <= 1, 'dataset ratio cannot be larger than 1'
            with open(triplet_path) as f:
                for line in tqdm(f, desc=f"Reading triplets from {triplet_path}"):
                    qid, pos_id, neg_id = line.strip().split("\t")
                    qid = header + qid
                    pos_id = header + pos_id
                    neg_id = header + neg_id
                    triplets.append((qid, pos_id, neg_id))
                    _query2pos[qid].add(pos_id)
                    _query2neg[qid].add(neg_id)
            qids = list(_query2pos.keys())
            if ratio < 1:
                indices = list(range(len(qids)))
                random.shuffle(indices)
                sample_num = int(len(qids) * ratio)
                qids = [qids[x] for x in indices[:sample_num]]
            for qid in qids:
                query2pos[qid] = list(_query2pos[qid])
                query2neg[qid] = list(_query2neg[qid])
    return triplets, query2pos, query2neg


def download_file(url, file_path):
    """
    Downloads file and store in a path (Code borrow from Sentence Transformers)
    Arguments
    ---------
    url: str
        link of the file
    path:    
        path to store the file
    """
    file_path = Path(file_path)
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print(
            f"Error when trying to download {url}. Response {req.status_code}",
            file=sys.stderr,
        )
        req.raise_for_status()
        return
    with open(file_path, "wb") as f:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()
