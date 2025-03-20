import random
from lsr.utils.dataset_utils import (
    read_collection,
    read_qrels,
    read_queries,
    read_triplets,
)
from torch.utils.data import Dataset
from collections import defaultdict


class MultipleNegativesWithLabel(Dataset):
    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        triplet_ids_path: str,
        train_group_size: int,
    ):
        docs_dict, doc_label_dict = read_collection(collection_path, classification_labels=True)
        _, self.query2pos, self.query2neg = read_triplets(triplet_ids_path)
        queries, query_labels = read_queries(queries_path, classification_labels=True)

        num_labels = len(list(doc_label_dict.values())[0])
        labels_mapping = []
        for i in range(num_labels):
            labels = sorted(set(
                [l[i] for l in doc_label_dict.values()] + [l[1][i] for l in query_labels]
            ))
            labels_mapping.append({l:idx for idx, l in enumerate(labels)})

        self.train_group_size = train_group_size
        self.q_dict = {qid: query for qid, query in queries if qid in self.query2pos}
        self.qids = list(self.q_dict.keys())
        self.docs_dict = docs_dict
        self.docs_label_dict = {
            doc: [labels_mapping[i][l] for i, l in enumerate(labels)] 
            for doc, labels in doc_label_dict.items()
        }
        self.q_label_dict = {
            qid: [labels_mapping[i][l] for i, l in enumerate(labels)] 
            for qid, labels in query_labels if qid in self.query2pos
        }
        # self.label2id = labels_mapping
        # self.id2label = [{i: label for label, i in mapping.items()} for mapping in labels_mapping] 

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, item):
        qid = self.qids[item]
        query = self.q_dict[qid]
        query_label = self.q_label_dict[qid]

        pos_id = random.choice(self.query2pos[qid])
        if len(self.query2neg[qid]) < self.train_group_size - 1:
            negs = random.choices(self.query2neg[qid], k=self.train_group_size - 1)
        else:
            negs = random.sample(self.query2neg[qid], k=self.train_group_size - 1)
        group_batch = [self.docs_dict[i] for i in [pos_id] + negs]
        batch_labels = [self.docs_label_dict[i] for i in [pos_id] + negs]

        return (query, query_label), list(zip(group_batch, batch_labels))
