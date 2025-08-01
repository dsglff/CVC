import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from models.GPT2_arch import AccustumGPT2Model
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from .prompt import Prompt 
from transformers import GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(32, hidden_dim)
        self.lineartime = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word_embedding = word_embedding.T

    def forward(self, x, x_time):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)
        x_time = self.lineartime(x_time)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)



        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        # self.input_len = self.patch_size * (configs.wavelet_j + 1)
        self.patch_num = int(((configs.seq_len - self.patch_size) // self.stride + 6 - 2 * self.patch_size / self.stride) / (2 ** configs.wavelet_j))
        # 小波变换配置
        decompose_layer = configs.wavelet_j
        wave = configs.wavelet
        self.prompt_length = configs.prompt_length
        mode = 'symmetric' 
        self.dwt = DWT1DForward(wave=wave, J=decompose_layer, mode=mode)  
        self.idwt = DWT1DInverse(wave=wave)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
                    
        self.task_name = configs.task_name
    
        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)
        
        self.word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        
        self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=True, pool_size=configs.pool_size, top_k=configs.prompt_length, batchwise_prompt=False, prompt_key_init=configs.prompt_init,wte = self.word_embedding)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])

        self.in_layer = Encoder_PCA(self.patch_size * (2 ** configs.wavelet_j), self.word_embedding, hidden_dim=configs.d_model)

        self.wte = self.gpt2.wte.state_dict()['weight'].cpu().numpy()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.attentionmaps_indices = [self.tokenizer.encode(word)[0] for word in self.attentionmaps]
        self.attentionmaps_vectors = self.wte[self.attentionmaps_indices]


        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model * self.patch_num * 2, configs.pred_len)
            self.out_layer_time = nn.Linear(configs.d_model * (self.patch_num + self.prompt_length), configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.out_layer_time, self.time_proj, self.text_proj, self.prompt_pool):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
        
    def get_patch(self, x):
        # x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # 
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #
        x = rearrange(x, 'b m n p -> (b m) n p') # 
        return x

    def forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        #计算小波系数
        in_dwt = x.permute(0,2,1)
        yl, yhs = self.dwt(in_dwt)
        coefs = [yl] + yhs

        last_dim_sizes = []
        for tensor in coefs:
            last_dim_sizes.append(tensor.size(-1))
        total = sum(last_dim_sizes)
        proportion_list = [element / total for element in last_dim_sizes]

        coefs_patch = []
        for tensor in coefs:
            a = self.get_patch(tensor)
            if a.size(1) == int(self.patch_num * 2):
                # x1, x2 = torch.split(a, 16, dim=1)
                x1 = a[:, ::2, :]  # 选择索引 [0, 2, 4] 的元素
                x2 = a[:, 1::2, :]  # 选择索引 [1, 3, 5] 的元素
                coefs_patch.append(x1)
                coefs_patch.append(x2)
            else:
                coefs_patch.append(a)
        
        time_patch = []
        for tensor in coefs:
            a = self.get_patch(tensor)
            time_patch.append(a)
        
        x_1 = torch.cat([time_patch[0], time_patch[2]], dim=1)
        x = torch.cat([x_1, time_patch[1]], dim=2)
        x_time = torch.cat(coefs_patch, dim=2)
        outputs_time1, outputs_text1 = self.in_layer(x, x_time)
        outs = self.prompt_pool(outputs_time1)
        prompted_embedding = outs['prompted_embedding']
        # sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']



        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=prompted_embedding)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        # outputs_time = outputs_time[:, self.prompt_length:, :]
        outputs_time = self.out_layer_time(outputs_time.reshape(B*M, -1))
        outputs_text = self.out_layer(outputs_text.reshape(B*M, -1))
        outputs_text = rearrange(outputs_text, '(b m) l -> b m l', b=B)
        outputs_time = rearrange(outputs_time, '(b m) l -> b m l', b=B)
        # 用于存储分割后的张量列表
        outputs_text_split = []
        outputs_time_split = []
        # 记录当前分割的起始索引
        start_index = 0
        last_dim_sizes = [int(proportion * outputs_text.size(-1)) for proportion in proportion_list]
        for size in last_dim_sizes:
            end_index = start_index + size
            split_tensor = outputs_text[..., start_index:end_index]
            outputs_text_split.append(split_tensor)

            split_tensor = outputs_time[..., start_index:end_index]
            outputs_time_split.append(split_tensor)
            start_index = end_index

        outputs_time_idwt = []
        outputs_text_idwt = []
        for i in range(len(outputs_text_split)):
            outputs_time_idwt.append(torch.cat((coefs[i], outputs_time_split[i]), 2))
            outputs_text_idwt.append(torch.cat((coefs[i], outputs_text_split[i]), 2))

        
        outputs_time = self.idwt((outputs_time_idwt[0], outputs_time_idwt[1:]))
        outputs_text = self.idwt((outputs_text_idwt[0], outputs_text_idwt[1:]))

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_time = outputs_time[:, -self.pred_len:, :]
        outputs_text = outputs_text[:, -self.pred_len:, :]


        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means



        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
            'simlarity_loss':simlarity_loss,
        }



    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
