import math
import torch
from torch import nn
from torch.autograd import Variable
from args_WUU import Path, Config
from simba import Block_mamba, Block_mamba_dct
# from simba import Block_mamba_dct as Block_mamba
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from args_WUU import Path, Config
from simba_DCT import Block_mamba, Block_mamba_dct, Block_mamba_init


import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import unbatch

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaVortex_TFV(nn.Module):
    
    def __init__(self, config):
        super(MambaVortex_TFV, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head, dim_feedforward=config.forward_hidden, dropout=config.dropout)
        # self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_1 = nn.ModuleList([Block_mamba_dct(dim=128, mlp_ratio=0.3, drop_path=0.2, cm_type='EinFFT') for _ in range(config.num_encoder)])

        # self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        # self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model)

        # self.position_multi = PositionalEncoding(d_model=config.dim_model, dropout=0.1)
        # encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head,dim_feedforward=config.forward_hidden, dropout=config.dropout)
        # self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)
        # self.transformer_encoder_multi = nn.ModuleList([Block_mamba_dct(dim=128, mlp_ratio=0.3, drop_path=0.2, cm_type='EinFFT') for _ in range(config.num_encoder_multi)])

        self.fc1 = nn.Sequential(
            nn.Linear(158, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, config.num_classes)
        )

        self.dct_layer = FcaBasicBlock(29, 29) # 29是时间维度


        self.hg1 = CosineHypergraphLayer2(128, 128, 30, 5)
        self.hg11 = CosineHypergraphLayer2(30, 30, 128, 10)

        self.hg2 = CosineHypergraphLayer2(128, 128, 30, 5)
        self.hg22 = CosineHypergraphLayer2(30, 30, 128, 10)

        self.gat1 = GATConv(config.dim_model, config.dim_model // config.gat_heads, heads=config.gat_heads, concat=True, dropout=config.dropout)
        self.gat2 = GATConv(config.dim_model, config.dim_model, heads=1, concat=False, dropout=config.dropout)
        self.gat3 = GATConv(config.dim_model, config.dim_model // config.gat_heads, heads=config.gat_heads, concat=True, dropout=config.dropout)
        self.gat4 = GATConv(config.dim_model, config.dim_model, heads=1, concat=False, dropout=config.dropout)

        self.gat11 = GATConv(30, 30 // 2, heads=2, concat=True, dropout=config.dropout)
        self.gat22 = GATConv(30, 30, heads=1, concat=False, dropout=config.dropout)
        self.gat33 = GATConv(30, 30 // 2, heads=2, concat=True, dropout=config.dropout)
        self.gat44 = GATConv(30, 30, heads=1, concat=False, dropout=config.dropout)

        
        # self.global_pool_time = nn.Linear(128,  2)
        # self.global_pool_freq = nn.Linear(29, 2)

        # self.fc_au = nn.Sequential(
        #     nn.Linear(config.pad_size * config.dim_model, config.fc_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(config.fc_hidden, config.num_classes_au)
        # )
        self.time_weight = nn.Parameter(torch.randn(10, 3))  # (out_time, group_size)
        self.freq_weight = nn.Parameter(torch.randn(32, 4))  


    def forward(self, x):
        x1 = x[:, 0, :, :]
        # x2 = x[:, 1, :, :]
        # x3 = x[:, 2, :, :]
        
        x1 = self.position_single(x1)
        # x2 = self.position_single(x2)
        # x3 = self.position_single(x3)

        for block in self.transformer_encoder_1:
            x1 = block(x1, 1, 29)     # (batch_size, 29, 128), (batch, time, frequency)
        # x2 = self.transformer_encoder_2(x2)
        # x3 = self.transformer_encoder_3(x3)

        # x = torch.cat([x1, x2, x3], dim=2)
        x = self.dct_layer(x1)

        x = self.drop(x)
        x = self.layer_norm(x)
        
        mean_row = x.mean(dim=1, keepdim=True) 
        x = torch.cat([x, mean_row], dim=1)
        residual = x

        
        ## time-wise evograph
        x, H1 = self.hg1(x)
        x = F.relu(x)
        residual_1 = x
        
        A_global, edge_index = hypergraph_to_adjacency(H1)

        edge_index = edge_index.to(x.device)
        data_list = []
        for i in range(x.shape[0]):
            node_feat = x[i]  
            data = Data(x=node_feat, edge_index=edge_index)
            data_list.append(data)
        batch_graph = Batch.from_data_list(data_list)
        out = F.elu(self.gat1(batch_graph.x, batch_graph.edge_index))
        out = F.elu(self.gat2(out, batch_graph.edge_index))
        
        node_features_list = unbatch(out, batch_graph.batch)
        out = torch.stack(node_features_list, dim=0)  # shape: (5, 30, 128)

        out = out.view(out.shape[0], 10, 3, 128)                # 分成10组，每组三个时间点
        w = self.time_weight.unsqueeze(-1)       # (10, 3, 1)
        out = (out * w).sum(dim=2)                 # (B, 10, 128)

        x, H2 = self.hg2(out)
        x = F.relu(x)
        A_global, edge_index = hypergraph_to_adjacency(H2)
        edge_index = edge_index.to(x.device)
        data_list = []
        for i in range(x.shape[0]):
            node_feat = x[i]  
            data = Data(x=node_feat, edge_index=edge_index)
            data_list.append(data)
        batch_graph = Batch.from_data_list(data_list)
        out = F.elu(self.gat3(batch_graph.x, batch_graph.edge_index))
        out = F.elu(self.gat4(out, batch_graph.edge_index))
        out = global_mean_pool(out, batch_graph.batch)

        ## frequency-wise evograph
        x_f = residual.permute(0, 2, 1)
        
        x_f, Hf = self.hg11(x_f)
        x_f = F.relu(x_f)
        res_1 = x_f
        
        A_global_f, edge_index_f = hypergraph_to_adjacency(Hf)

        edge_index_f = edge_index_f.to(x_f.device)
        data_list_f = []
        for i in range(x_f.shape[0]):
            node_feat_f = x_f[i]  
            data = Data(x=node_feat_f, edge_index=edge_index_f)
            data_list_f.append(data)
        batch_graph_f = Batch.from_data_list(data_list_f)
        out_f = F.elu(self.gat11(batch_graph_f.x, batch_graph_f.edge_index))
        out_f = F.elu(self.gat22(out_f, batch_graph_f.edge_index))

        node_features_list_f = unbatch(out_f, batch_graph_f.batch)
        out_f = torch.stack(node_features_list_f, dim=0)

        out_f = out_f.view(out_f.shape[0], 32, 4, 30)                # 分成10组，每组三个时间点
        w_f = self.freq_weight.unsqueeze(-1)    
        out_f = (out_f * w_f).sum(dim=2)                 # (B, 10, 128)

        x_f, Hf = self.hg22(x_f)
        x_f = F.relu(x_f)
        res_1 = x_f
        
        A_global_f, edge_index_f = hypergraph_to_adjacency(Hf)

        edge_index_f = edge_index_f.to(x_f.device)
        data_list_f = []
        for i in range(x_f.shape[0]):
            node_feat_f = x_f[i]  
            data = Data(x=node_feat_f, edge_index=edge_index_f)
            data_list_f.append(data)
        batch_graph_f = Batch.from_data_list(data_list_f)
        out_f = F.elu(self.gat33(batch_graph_f.x, batch_graph_f.edge_index))
        out_f = F.elu(self.gat44(out_f, batch_graph_f.edge_index))
        out_f = global_mean_pool(out_f, batch_graph_f.batch)

        ## auxilary classifier By WUU
        # x_time = self.global_pool_time(x)
        # x_freq = x.permute(0, 2, 1)
        # x_freq = self.global_pool_freq(x_freq)
        # x_freq = x_freq.permute(0, 2, 1)
        # x_au = torch.bmm(x_time, x_freq)
        # x_au = self.drop(x_au)
        # x_au = x_au.view(x_au.size(0), -1)
        # x_au_residual = x_au
        # x_au = self.fc_au(x_au)
        ## auxilary classifier By WUU

        x = torch.cat((out, out_f), dim=-1)

        x = x.view(x.size(0), -1) # 这里增加辅助分类器的特征
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    config = Config()
    x = torch.rand(5, 1, 29, 128).to(config.device)
    m = MambaVortex_TFV(config).to(config.device)
    y = m(x)
    # y, y_au = m(x)
    print(y.data.cpu().numpy().shape)
    # print(y_au.data.cpu().numpy().shape)

    from thop import profile
    from thop import clever_format

    m = MambaVortex_TFV(config).to(device='cuda:0')
    input_tensor = torch.randn(64, 3, 29, 128).to(device='cuda:0')
    flops, params = profile(m, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")