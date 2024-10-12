import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import networkx as nx

import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from build_vocab import WordVocab
from utils import *

from dataset import DTADataset
from sklearn.model_selection import KFold
from model import *
from torch import nn as nn


#############################################################################

CUDA = '0'
device = torch.device('cuda:'+CUDA)
LR = 1e-3
NUM_EPOCHS = 200
seed = 0
batch_size = 128
dataset_name = 'davis'


#############################################################################

class DMFF(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5):
        super(DMFF, self).__init__()
        self.is_bidirectional = True
        # drugs
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads
        self.MGNN = GINConvNet(num_features_xd = lstm_dim * 2 + 1, n_output=hidden_dim * 2)
    
        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance1= SpatialGroupEnhance_for_1D(groups=20)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.is_bidirectional = True
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)
   
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)

        self.fusion_graph_seq = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

        # Point-wise Feed Forward Network
        self.pwff_1 = nn.Linear(hidden_dim * 3, hidden_dim * 4)
        self.pwff_2 = nn.Linear(hidden_dim * 4, hidden_dim * 3)
   
    def forward(self, data, reset=False):
        batchsize = len(data.sm)
        smiles = torch.zeros(batchsize, seq_len).to(device).long()
        protein = torch.zeros(batchsize, tar_len).to(device).long()
        smiles_lengths = []
        protein_lengths = []


        smiles = data.smiles.to(device)
        protein = data.protein.to(device)
        smiles_lengths = data.smiles_lengths
        protein_lengths = data.protein_lengths


        smiles = smiles.view(batchsize, -1)
        protein = protein.view(batchsize, -1)


        # for i in range(batchsize):
        #     sm = data.sm[i]
        #     seq_id = data.target[i]
        #     seq = target_seq[seq_id]
        #     smiles[i] = smiles_emb[sm]
        #     protein[i] = target_emb[seq]
        #     smiles_lengths.append(smiles_len[sm])
        #     protein_lengths.append(target_len[seq])

        smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim
        smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        smiles = self.enhance1(smiles)


        protein = self.protein_embed(protein)  # B * tar_len * emb_dim
        protein = self.protein_input_fc(protein)  # B * tar_len * lstm_dim
        protein = self.enhance2(protein)
   


        # drugs and proteins BiLSTM
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        smiles = self.ln1(smiles)
        protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        protein = self.ln2(protein)

        if reset:
            return smiles, protein

        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len
        
        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)  # B * head * tar_len


        smiles_out, smile_attn = self.out_attentions3(smiles, smiles_mask)  # B * lstm_dim*2
        protein_out, prot_attn = self.out_attentions2(protein, protein_mask)  # B * (lstm_dim *2)



        # drugs and proteins
        out_cat = torch.cat((smiles, protein), dim=1)  # B * head * lstm_dim *2
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * tar_len+seq_len * (lstm_dim *2)
        out_cat, out_attn = self.out_attentions(out_cat, out_masks)
        out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)  # B * (rnn*2 *3)

        # Point-wise Feed Forward Network
        pwff = self.pwff_1(out)
        pwff = nn.ReLU()(pwff)
        pwff = self.dropout(pwff)  
        pwff = self.pwff_2(pwff)
        
        out = pwff + out 


        out = self.dropout(self.relu(self.out_fc1(out)))  # B * (256*8)
        out = self.dropout(self.relu(self.out_fc2(out)))  # B *  hidden_dim*2

        gout = self.MGNN(data)
        out = torch.cat([gout, out], dim=-1)  # B * (hidden_dim*4)
        out = self.dropout(self.relu(self.fusion_graph_seq(out)))  # B * (hidden_dim*2)

        out = self.out_fc3(out).squeeze()

        del smiles_out, protein_out

        return out

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return c_size, edge_index

#############################################################################



df = pd.read_csv(f'./{dataset_name}_processed.csv')

smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])

target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']

smiles_graph = {}
for sm in smiles:
    _, graph = smiles_to_graph(sm)
    smiles_graph[sm] = graph


target_uniprot_dict = {}
target_process_start = {}
target_process_end = {}

for i in range(len(df)):
    target = df.loc[i,'target_key']
    if dataset_name == 'kiba':
        uniprot = df.loc[i,'target_key']
    else:
        uniprot = df.loc[i,'uniprot']
    target_uniprot_dict[target] = uniprot
    target_process_start[target] = df.loc[i,'target_sequence_start']
    target_process_end[target] = df.loc[i,'target_sequence_end']


contact_dir = './target_contact_map_' + dataset_name + '/'
target_graph = {}
def target_to_graph(target_key, target_sequence, contact_dir,start,end):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map = contact_map[start:end, start:end]
    index_row, index_col = np.where(contact_map > 0.8)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_edge_index
        

for target in tqdm(target_seq.keys()):
    uniprot = target_uniprot_dict[target]
    contact_map = np.load(contact_dir + uniprot + '.npy')
    start = target_process_start[target]
    end = target_process_end[target]
    _, graph = target_to_graph(uniprot, target_seq[target], contact_dir,start,end)
    target_graph[target] = graph


drug_vocab = WordVocab.load_vocab('./Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('./Vocab/protein_vocab.pkl')

tar_len = 1000
seq_len = 540

smiles_idx = {}
smiles_emb = {}
smiles_len = {}
for sm in smiles:
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag = flag + 1

    if len(content) > seq_len:
        content = content[:seq_len]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem


target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    if len(content) > tar_len:
        content = content[:tar_len]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)

print("Building dataset...")
dataset = DTADataset(root='./', path='./'+ dataset_name+ '_processed.csv', smiles_emb=smiles_emb, target_emb=target_emb, smiles_idx=smiles_idx, smiles_graph=smiles_graph, target_graph=target_graph, smiles_len=smiles_len, target_len=target_len)


def reset_feature(dataset , model):
    torch.cuda.empty_cache()
    batch_size = 128
    with torch.no_grad():
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        start = 0
        for data in tqdm(dataloader):
            sm , pro = model(data,reset = True)
            tar_len = []
            idx = []
            for i in range(min(batch_size,len(dataset) - start)):
                sm_id = dataset[start+i].sm
                pro_id = dataset[start+i].target
                pro_id = target_seq[pro_id]
                tar_len.append(target_len[pro_id])
                idx.append(smiles_idx[sm_id])
            for i in range(start , min(len(dataset) , start + batch_size)):
                total_len = tar_len[i-start] + len(idx[i-start]) + 1 
                source = torch.zeros(total_len, dtype=torch.long).to(device)
                source[:tar_len[i-start]] = 0 
                source[tar_len[i-start]:tar_len[i-start]+len(idx[i-start])] = 1
                source[-1] = 2
                feature = torch.cat([pro[i-start,1:tar_len[i-start]+1] , sm[i-start,idx[i-start]] , (sm[i-start,0].unsqueeze(0) + pro[i-start,0].unsqueeze(0))/2])
                
                new_feature = torch.cat([feature , source.unsqueeze(-1)],dim=-1) 
                dataset.data[i].x = new_feature
            start = start+batch_size


dataset_name = f"{dataset_name}_processed"
model_name = 'default'

model_file_name = './Model/'+dataset_name+'_'+model_name+'.pt'    

load_model_path = model_file_name



num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True,random_state=0)

for fold, (train_indices, test_indices) in enumerate(kf.split(dataset)):
    print("Building model...")
    model = DMFF(embedding_dim=256, lstm_dim=128, hidden_dim=256, dropout_rate=0.2,
                    alpha=0.2, n_heads=8, bilstm_layers=2, protein_vocab=26,
                    smile_vocab=45, theta=0.5).to(device)

    # load model
    if load_model_path is not None:
        save_model = torch.load(load_model_path)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-4, last_epoch=-1)
    
    
    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1
    print(f"Fold {fold+1}")

    for epoch in range(NUM_EPOCHS):
        print("No {} epoch".format(epoch))
        if epoch == 0:
            print('reset feature')
            reset_feature(dataset , model)
            val_size = int(len(dataset) * 0.1)
            train_dataset = torch.utils.data.Subset(dataset, train_indices)

            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset=train_dataset,
                lengths=[len(train_dataset)-val_size , val_size],
                generator=torch.Generator().manual_seed(0)
            )

            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train(model , train_loader,optimizer,epoch)
        G, P = predicting(model, val_loader)
        val1 = get_mse(G, P)
        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            if model_file_name is not None:
                torch.save(model.state_dict(), model_file_name)
            print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
        else:
            print('current mse: ',val1 ,  ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)
        schedule.step()
    print(model_file_name)
    save_model = torch.load(model_file_name)
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    G, P = predicting(model, test_loader)
    cindex,rm2,mse= calculate_metrics_and_return(G, P, test_loader)
    break
        
