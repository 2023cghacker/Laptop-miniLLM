import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from dataclasses import dataclass
import torch.nn.functional as F
from torch.utils.data import Dataset
import tiktoken
import json  # 添加这个导入
from torch.utils.data import DataLoader


@dataclass
class GPTConfig:
    # 配置
    block_size: int = 512  # context length, max sequence length
    batch_size: int = 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384  # hidden dimension of the model
    dropout: float = 0.1

    head_dim: int = 32  # dimension of each attention head
    vocab_size: int = 50274  # vocabulary size of the model (gpt2)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SingleHeadAttention(nn.Module):
    # 单头注意力
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_dim)
        self.query = nn.Linear(config.n_embd, config.head_dim)
        self.value = nn.Linear(config.n_embd, config.head_dim)
        self.head_dim = config.head_dim

        # attention_mask通过register_buffer来注册，因为attention_mask不会参与梯度更新
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)),  # 下三角矩阵
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(
            -2, -1
        )  # 计算注意力权重，@表示矩阵乘法，transpose(-2,-1)表示交换最后两个维度
        weight = weight.masked_fill(  # masked_fill是torch.Tensor的一个方法，用于将weight中满足条件的元素填充为-inf
            self.attention_mask[:seq_len, :seq_len]
            == 0,  # 将attention_mask中为0的元素填充为-inf
            float("-inf"),
        )
        # np.sqrt(self.head_dim) 是计算注意力权重的缩放因子，用于防止梯度消失或爆炸
        weight = torch.softmax(weight / np.sqrt(self.head_dim), dim=-1)  # 归一化
        weight = self.dropout(weight)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    # 多头注意力
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList(  # 将多个单头注意力拼接起来
            [SingleHeadAttention(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # 前馈神经网络
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        out = self.net(x)
        out = self.norm(out)
        return out


class Block(nn.Module):
    # 一个transformer块
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.mha = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    # GPT模型
    def __init__(self, config: GPTConfig):
        super().__init__()
        # nn.Embedding是pytorch中的一个类，用于将离散的整数索引转换为连续的向量表示
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # bias=False: 因为lm_head的bias会影响模型的输出
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 将token_embedding的权重与lm_head的权重共享(tie weights)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)  # 初始化参数权重

    def _init_weights(self, module):
        # 初始化为正态分布
        if isinstance(module, nn.Linear):  # 如果module是nn.Linear类型
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):  # 如果module是nn.Embedding类型
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: 输入的token序列
        targets: 目标token序列
        """
        batch_size, seq_len = idx.shape
        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(
            torch.arange(
                seq_len, device=idx.device
            )  # arange: 生成一个从0到seq_len-1的序列
        )
        x = token_emb + position_emb
        x = self.blocks(x)  # 经过多个transformer块
        x = self.ln_f(x)  # 这里是layer norm
        logits = self.lm_head(x)  # 输出logits

        if targets is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            # 将logits展平
            logits = logits.view(batch_size * seq_len, vocab_size)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):  # 生成max_new_tokens个token
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # 取最后一个token的logits
            probs = F.softmax(logits, dim=-1)  # 归一化
            idx_next = torch.multinomial(probs, num_samples=1)  # 随机采样
            idx = torch.cat((idx, idx_next), dim=-1)  # 将新的token加入到idx中

        return idx


class MyDataset(Dataset):
    def __init__(self, data_path, block_size=512):
        # 我的路径在data/webtext.train.jsonl
        # 每一行的格式是{"id": 0, "ended": true, "length": 138, "text": "These girlfriends deserves a special mention for going that extra mile, hopefully doesn't set too many guys off on the path towards outrageous demands.\n\n1. She knows the severity of man-flu\n\n2. All fun and games is all good\n\n3. A voucher that says 'I love you'\n\n4. When arguments don't drag on forever.\n\n5. Providing everything he needs.\n\n6. Very understanding\n\n7. As awesome a gesture as this is, we are worried about this man's cooking skills.\n\n8. Nice cake\n\n8. Fair bargaining\n\n9. Excellent gift choice\n\n10. Very thoughtful"}
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.max_lines = 1000
        self.encoded_data = []

        raw_data = []
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    text = data["text"]
                    raw_data.append(text)
                except Exception as e:
                    print(f"Error processing line {i}: {e}")
                    continue

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.enc.eot_token])

        # block=512,需要将长文本分割为多个block
        for idx in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[idx : idx + self.block_size + 1]  # 513个
            if len(chunk) < self.block_size + 1:
                continue  # 丢弃

            else:
                self.encoded_data.append(chunk)

        # 最后得到的encoded_data是一个列表,大小为(n,block_size)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, tokens):
        return self.enc.decode(tokens)


def draw_progress_bar(x, n, loss, bar_length=50):

    # 计算进度条的填充比例
    progress_ratio = x / n
    filled_length = int(bar_length * progress_ratio)
    empty_length = bar_length - filled_length

    # 绘制进度条
    filled_bar = "█" * filled_length  # 使用"█"表示已填充部分
    empty_bar = " " * empty_length  # 使用空格表示未填充部分

    # 打印进度条
    print(f"\r|{filled_bar}{empty_bar}| current batch={x}/{n} loss={loss}", end="")

    # 如果进度完成，换行
    if x == n:
        print()


if __name__ == "__main__":

    # 1. 定义参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(GPTConfig())
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    # 设置scheduler: 余弦退火, 学习率从0.0005到0.0001
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    dataset = MyDataset("train_text.jsonl")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(
        f"\nDevice: {device}"
        f"\nTotal parameters: {total_params}"
        f"\nlen(dataset): {len(dataset)}\n"
    )

    # train
    model.train()
    for epoch in range(10):
        print(f"\n[training] current epoch={epoch}")
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()

            draw_progress_bar(batch_idx + 1, len(dataloader), loss.item())
            # print("loss=", loss.item())

        # vaildation: 给一个句子，输出其预测的下一个单词
        model.eval()
        x = "i want to be a good"
        print("\n[vaildation]\ninput text: ", x)
        x = dataset.encode(x)
        x = torch.tensor(x, dtype=torch.long)
        x = x.unsqueeze(0)  # 将x大小变成（1，n)
        x = x.to(device)
        new_x = model.generate(x, 20)  # 接着续写这个句子
        print("output text: ", dataset.decode(new_x[0].tolist()))
        model.train()

        # save model (分批次）
        torch.save(model.state_dict(), f"model_{epoch}.pt")
