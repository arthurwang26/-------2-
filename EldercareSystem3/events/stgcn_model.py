import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Graph:
    def __init__(self, strategy='spatial'):
        # COCO 17 keypoints
        # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow,
        # 9:LWrist, 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        self.num_node = 17
        self.edges = [
            (0, 1), (0, 2), (1, 3), (2, 4), # Head
            (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12) # Torso & Legs
        ]
        self.center = 0 # Nose as root
        self.A = self.get_adjacency_matrix(strategy)

    def get_adjacency_matrix(self, strategy):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        for i in range(self.num_node):
            A[i, i] = 1

        if strategy == 'spatial':
            # 3 partitions: root(self), centripetal(closer to center), centrifugal(further from center)
            A_out = np.zeros((3, self.num_node, self.num_node))
            
            # Compute shortest distance from center to all nodes (BFS)
            dist = np.full(self.num_node, np.inf)
            dist[self.center] = 0
            queue = [self.center]
            while queue:
                u = queue.pop(0)
                for v in range(self.num_node):
                    if A[u, v] == 1 and dist[v] == np.inf:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if A[i, j] == 1:
                        if i == j:
                            A_out[0, i, j] = 1 # Root
                        elif dist[j] < dist[i]:
                            A_out[1, i, j] = 1 # Centripetal
                        else:
                            A_out[2, i, j] = 1 # Centrifugal
            
            # Normalize
            for k in range(3):
                row_sum = A_out[k].sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1 # prevent division by zero
                A_out[k] = A_out[k] / row_sum
            
            return torch.tensor(A_out, dtype=torch.float32)
        else:
            raise ValueError()

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 
                              kernel_size=(t_kernel_size, 1), 
                              padding=(t_padding, 0), 
                              stride=(t_stride, 1), 
                              dilation=(t_dilation, 1))

    def forward(self, x, A):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        x = self.conv(x) # (N, C' * K, T, V)
        x = x.view(N, self.kernel_size, -1, T, V) # (N, K, C', T, V)
        
        # A: (K, V, V)
        # We want to multiply x (N, K, C', T, V) with A (K, V, V) over the V dimension
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()
        
        spatial_kernel = kernel_size[1]
        temporal_kernel = kernel_size[0]
        padding = ((temporal_kernel - 1) // 2, 0)
        
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, spatial_kernel)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1),
                      padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)

class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, graph_args={}, edge_importance_weighting=True):
        super().__init__()
        
        # Load Graph
        self.graph = Graph(**graph_args)
        A = self.graph.A
        self.register_buffer('A', A)
        
        spatial_kernel = A.size(0)
        temporal_kernel = 9
        kernel_size = (temporal_kernel, spatial_kernel)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        # Lightweight ST-GCN Architecture (faster, prevents overfitting on small datasets)
        self.st_gcn_networks = nn.ModuleList([
            STGCN_Block(in_channels, 64, kernel_size, 1, residual=False),
            STGCN_Block(64, 64, kernel_size, 1),
            STGCN_Block(64, 64, kernel_size, 1),
            STGCN_Block(64, 128, kernel_size, 2), # stride 2 reduces temporal dim by half
            STGCN_Block(128, 128, kernel_size, 1),
            STGCN_Block(128, 256, kernel_size, 2),
            STGCN_Block(256, 256, kernel_size, 1),
        ])
        
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        
        # Data Normalization
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
            
        # Global pooling
        # x: (N, C', T', V) -> (N, C', 1, 1)
        x = F.avg_pool2d(x, x.size()[2:])
        
        # Prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x
