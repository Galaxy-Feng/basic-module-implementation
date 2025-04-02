import torch
from torch import nn


def masked_softmax(X, valid_lens):

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        # Create a mask for invalid positions
        mask = torch.arange(X.size(1), device=X.device).expand_as(X) >= valid_lens.unsqueeze(-1)
        X = torch.masked_fill(X, mask, -1e6)
        return nn.functional.softmax(X, dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size: int, query_size: int, num_hiddens: int, dropout: float, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, valid_lens: torch.Tensor):

        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# Example usage
queries = torch.normal(0, 1, (2, 1, 20))
keys = torch.ones((2, 10, 2))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
output = attention(queries, keys, values, valid_lens)
print(output)