import torch

class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, group_size):
        super(GroupedQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embeddings into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )
        out = self.fc_out(out)
        return out


embed_size = 256  # 嵌入维度
num_heads = 8     # 头的数量
group_size = 4    # 每个头的分组大小

gqa = GroupedQueryAttention(embed_size, num_heads, group_size)
# 一个batch的大小为1，序列长度为10的输入query
query = torch.randn(1, 10, embed_size)
# 一个batch的大小为1，序列长度为20的输入key
key = torch.randn(1, 20, embed_size)
# 一个batch的大小为1，序列长度为20的输入value
value = torch.randn(1, 20, embed_size)

# 运行前向传播
output = gqa(value, key, query, mask=None) # 假设我们没有mask，所以传递None
print(output)
