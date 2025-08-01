import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

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





class Prompt(nn.Module):
    def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=30, top_k=4, batchwise_prompt=False, prompt_key_init='uniform',wte = None):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        print(self.pool_size)
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.wte = wte
        self.linear  = nn.Linear(768, 512)
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(key_shape),requires_grad=False)
                print('zero initialized key')
                
            elif prompt_key_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.uniform_(self.prompt, -5, 5)
                print('uniform initialized key')
            
            elif prompt_key_init == 'gaussian':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.normal_(self.prompt, mean=0.0, std=5.0)
                print('gaussian initialized key')

            elif prompt_key_init == 'text_prototype':
                self.text_prototype_linear = nn.Linear(50257, pool_size)
                
          
        





        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def get_prompt(self, x_embed_norm, prompt_norm):
    # 计算相似性并构建图
    # 示例向量矩阵
        # 计算余弦相似度
        node_features = torch.cat([x_embed_norm, prompt_norm], dim=0)
        similarity = torch.matmul(node_features, node_features.transpose(-1, -2))  
        # 构建稀疏邻接矩阵
        threshold = 0.0  # 仅保留相似度高于阈值的边
        adj_matrix = (similarity > threshold).float()  

        # 边索引
        edge_index = adj_matrix.nonzero(as_tuple=True)  # 提取非零边（邻接关系）
        # 创建图数据
        graph_data = Data(x=node_features, edge_index=torch.stack(edge_index, dim=0))


        # 初始化模型
        model = GNN(in_channels=node_features.shape[1], out_channels=768).to(similarity.device)
        # 前向传播
        output_features = model(graph_data)

        # 4. 提取邻近向量
        # 提取 A 和 B 对应的节点嵌入
        A_embeds = output_features[:x_embed_norm.shape[0]]
        B_embeds = output_features[x_embed_norm.shape[0]:]
        # 再次计算余弦相似度
        final_similarity = F.cosine_similarity(A_embeds.unsqueeze(1), B_embeds.unsqueeze(0), dim=2)
        # final_similarity = torch.matmul(A_embeds, B_embeds.T)
        # 寻找最近邻
        _, nearest_indices = torch.topk(final_similarity, k=self.top_k, dim=1)  # 每个 A 的向量找到最相似的 B 的向量索引
        return nearest_indices, final_similarity


    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_key:   #if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            
            if self.prompt_key_init == 'text_prototype':
                # prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)
                prompt_key = self.wte.transpose(0, 1)
            else:
                prompt_key = self.prompt
            prompt_norm = self.l2_normalize(prompt_key, dim=1) # Pool_size, C   self.prompt_key
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
            # similarity = torch.matmul(x_embed_norm, prompt_norm.transpose(-1, -2)) # B, Pool_size
            # if prompt_mask is None:
            #     _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            #     if self.batchwise_prompt:
            #         prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            #         # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
            #         # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
            #         # Unless dimension is specified, this will be flattend if it is not already 1D.
            #         if prompt_id.shape[0] < self.pool_size:
            #             prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
            #             id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            #         _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
            #         major_prompt_id = prompt_id[major_idx] # top_k
            #         # expand to batch
            #         idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            # else:
            #     idx = prompt_mask # B, top_k

#
            idx, final_similarity = self.get_prompt(x_embed_norm, prompt_norm)

            # batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
                
            batched_prompt_raw = prompt_key[idx] # B, top_k, length, C
            batched_prompt_raw = batched_prompt_raw.unsqueeze(2) # B, top_k, 1, length, C

            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            # out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompt_key'] = prompt_key  # prompt_key

        return out