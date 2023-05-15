import os
from random import shuffle

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.dataloader import GenericDataLoader


class Head(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=model_dim[0], out_features=model_dim[1])
        self.fc2 = nn.Linear(in_features=model_dim[1], out_features=model_dim[1])
        self.fc3 = nn.Linear(in_features=model_dim[1], out_features=model_dim[2])

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


class Retrival_model():

    def __init__(self, downstream_head, llm, tokenizer, data_path, optimizer, loss_fn, sim_fun, save_path, device):

        self.device = device
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.sim_fun = sim_fun

        self.llm = llm.to(device)
        self.downstream_head = downstream_head.to(device)
        self.corpus, self.queries, self.qrels = GenericDataLoader(data_folder=data_path).load(split="train")
        self.corpus_test, self.queries_test, self.qrels_test = GenericDataLoader(data_folder=data_path).load(split="test")

        self.emb_BERT, self.corps_index = self.get_BERT_embs()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        encoded_input = {k: v for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.llm(**encoded_input)
        return self.mean_pooling(model_output, encoded_input["attention_mask"])

    def test_rank(self, ):
        emb_corpus = torch.Tensor([]).to(self.device)
        for emb in torch.split(self.emb_BERT, split_size_or_sections=1000, dim=0):
            with torch.no_grad():
                emb_corpus_itr = self.downstream_head(emb)
            emb_corpus = torch.cat([emb_corpus, emb_corpus_itr], dim=0)

        emb_qrels = torch.Tensor([]).to(self.device)

        query_index = list(self.qrels_test.keys())
        for query_array in np.array_split(query_index, 100):
            query = []
            for query_id in query_array:
                query.append(self.queries_test[query_id])

            emb_query = self.get_embeddings(query)
            with torch.no_grad():
                emb_query = self.downstream_head(emb_query)
            emb_qrels = torch.cat([emb_qrels, emb_query])

        sim = self.sim_fun(emb_qrels, emb_corpus.t())

        indexes = torch.argsort(sim, dim=1, descending=True)

        corps_idx_arr = [int(i) for i in self.corps_index]
        corps_idx_arr = torch.Tensor(corps_idx_arr).to(self.device)

        idx = []
        for i in range(indexes.shape[0]):
            ord = corps_idx_arr[indexes[i]]
            for cor in self.qrels_test[query_index[i]].keys():
                idx.append(1 / ((ord == int(cor)).nonzero(as_tuple=True)[0].cpu().item() + 1))

        print("Mean Reciprocal Rank", np.mean(idx))

        return np.mean(idx)

    def get_BERT_embs(self, ):
        emb_BERT = torch.Tensor([]).to(self.device)
        corps_index = list(self.corpus.keys())
        for corps_array in np.array_split(corps_index, 100):
            corps = []
            for corpus_id in corps_array:
                corps.append(self.corpus_test[corpus_id]['text'])

            emb_BERT_itr = self.get_embeddings(corps)
            emb_BERT = torch.cat([emb_BERT, emb_BERT_itr])
        print("Emb calculated")
        return emb_BERT, corps_index

    def train(self, epochs):
        keys = list(self.qrels.keys())

        MRR = []
        loss_tot = []
        for epoch in range(epochs):
            tot_loss = 0
            itr = 0
            shuffle(keys)
            for k in tqdm(np.array_split(keys, 10), desc="Epoch " + str(epoch)):
                itr += 1
                self.optimizer.zero_grad()
                q = []
                for k_ in k:
                    q.append(self.queries[k_])
                emb_query = self.get_embeddings(q)

                emb_query = self.downstream_head(emb_query)

                true_idx = []

                for k_ in k:
                    corpus_id = list(self.qrels[k_].keys())[0]
                    true_idx.append(self.corps_index.index(corpus_id))

                emb_corpus = torch.Tensor([]).to(self.device)
                for emb in torch.split(self.emb_BERT, split_size_or_sections=1000, dim=0):
                    with torch.no_grad():
                        emb_corpus_itr = self.downstream_head(emb)
                    emb_corpus = torch.cat([emb_corpus, emb_corpus_itr], dim=0)

                true_idx = torch.Tensor(true_idx).to(self.device).type(torch.int64)

                sim = self.sim_fun(emb_query, emb_corpus.t())

                loss = self.loss_fn(sim, true_idx)

                tot_loss += loss.detach().cpu().item()
                loss.backward()
                self.optimizer.step()

            print("Epoch", epoch, "Train Loss", tot_loss / itr, )
            loss_tot.append(tot_loss / itr)

            if epoch % 10 == 0:
                MRR.append(self.test_rank())
        plt.figure(1)
        plt.plot(range(0, epochs, 10), MRR)
        plt.ylabel("Mean Reciprocal Rank progress in training")
        plt.xlabel("Epochs")
        plt.title("Mean Reciprocal Rank")
        plt.savefig(os.path.join(self.save_path, "mrr"))

        plt.figure(2)
        plt.plot(range(epochs), loss_tot)
        plt.title("Loss progress in training")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.save_path, "loss"))
