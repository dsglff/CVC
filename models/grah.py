import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 3. 使用 GNN 学习节点关系
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def get_prompt(A, B):
# 计算相似性并构建图
# 示例向量矩阵
    A = torch.randn(10, 16)  # 10 个 16 维向量
    B = torch.randn(15, 16)  # 15 个 16 维向量
    # 计算余弦相似度
    similarity = torch.matmul(A, B.T)  # 余弦相似度矩阵，形状为 [10, 15]
    # 构建稀疏邻接矩阵
    threshold = 0.8  # 仅保留相似度高于阈值的边
    adj_matrix = (similarity > threshold).float()  # 二值邻接矩阵

    # 2. 用 PyTorch Geometric 定义图
    # 构建节点特征
    node_features = torch.cat([A, B], dim=0)  # 合并两个矩阵的向量作为节点特征
    # 边索引
    edge_index = adj_matrix.nonzero(as_tuple=True)  # 提取非零边（邻接关系）
    # 创建图数据
    graph_data = Data(x=node_features, edge_index=torch.stack(edge_index, dim=0))


    # 初始化模型
    model = GNN(in_channels=node_features.shape[1], out_channels=16)
    # 前向传播
    output_features = model(graph_data)

    # 4. 提取邻近向量
    # 提取 A 和 B 对应的节点嵌入
    A_embeds = output_features[:A.shape[0]]
    B_embeds = output_features[A.shape[0]:]
    # 再次计算余弦相似度
    final_similarity = torch.matmul(A_embeds, B_embeds.T)
    # 寻找最近邻
    _, nearest_indices = torch.topk(final_similarity, k=1, dim=1)  # 每个 A 的向量找到最相似的 B 的向量索引
    return nearest_indices


