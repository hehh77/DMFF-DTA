from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader, Dataset

import re
class DTADataset(InMemoryDataset):
    def __init__(self, root, path, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):

        super(DTADataset, self).__init__(root)
        self.path = path
        df = pd.read_csv(path)
        self.data = []
        self.process(df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len)



    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):
        # data_list = []
        for i in tqdm(range(len(df))):
            sm = df.loc[i, 'compound_iso_smiles']
            target = df.loc[i, 'target_key']
            seq = df.loc[i, 'target_sequence']
            label = df.loc[i, 'affinity']
            sm_g = smiles_graph[sm]
            ta_g = target_graph[target]
            sm_idx = smiles_idx[sm]
            tar_len = target_len[seq]

            s_off = self.off_adj(sm_g, tar_len)
            com_adj = np.concatenate((ta_g, s_off), axis=0)
            total_len = tar_len + len(sm_idx)
            tem1 = np.zeros([total_len, 2])
            tem2 = np.zeros([total_len, 2])
            for i in range(total_len):
                tem1[i, 0] = total_len
                tem1[i, 1] = i
                tem2[i, 1] = total_len
                tem2[i, 0] = i
            tem1 = np.int64(tem1)
            tem2 = np.int64(tem2)
            com_adj = np.concatenate((com_adj, tem1), axis=0)
            com_adj = np.concatenate((com_adj, tem2), axis=0)
            com_adj = np.concatenate((com_adj, [[total_len, total_len]]), axis=0)

            smiles = smiles_emb[sm]
            protein = target_emb[seq]
            smiles_lengths = smiles_len[sm]
            protein_lengths = target_len[seq]

            # smiles[i] = smiles_emb[sm]
            # protein[i] = target_emb[seq]
            # smiles_lengths.append(smiles_len[sm])
            # protein_lengths.append(target_len[seq])
            Data = DATA(y=torch.FloatTensor([label]),
                        edge_index=torch.LongTensor(com_adj).transpose(1, 0),
                        sm=sm,
                        target=target,
                        smiles=smiles,
                        protein=protein,
                        smiles_lengths=smiles_lengths,
                        protein_lengths=protein_lengths,
                        seq = seq
                        )
            self.data.append(Data)
        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(self.data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]


    def off_adj(self, adj, size):
        adj1 = adj.copy()
        for i in range(adj1.shape[0]):
            adj1[i][0] += size
            adj1[i][1] += size
        return adj1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]