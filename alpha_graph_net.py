"""ALPHA graph encoder (legacy_alpha/net.py), vectorised over (batch x ego agent).

Per ego agent ALPHA computes (legacy ALPHANet.forward, loop body):
    node_emb  = static_embedding(graph_nodes[ego])            [K, E]
    agent_emb = dynamic_embedding(agent_intent)                [N, E]
    w_node  = softmax(q(node_emb[node_index]) . k(node_emb)^T / sqrt(E))    (FocusAttention)
    w_agent = softmax(q(agent_emb[ego])       . k(agent_emb)^T / sqrt(E))   (same weights)
    node_emb, agent_emb = w_node * node_emb, w_agent * agent_emb
    node_emb, agent_emb = EncoderLayer(node_emb), EncoderLayer(agent_emb)   (shared layer)
    graph_feature = node_agent_embedding([node_emb[node_index], agent_emb[ego]])  -> NET_SIZE

Exact implementation changes (same function, see README):
  * the python loop over ego agents is flattened into the batch dimension (every op acts
    on one ego's sequences only);
  * agent embeddings and agent focus-attention scores are computed once per sample and
    shared by all egos (legacy embedded N identical copies);
  * focus-attention scores use (q Wq) Wk^T h^T instead of (q Wq)(h Wk)^T (associativity);
  * the last encoder layer only evaluates the query / feed-forward path for the ego
    token, the only token ALPHA reads from its output (keys/values still use all tokens).
"""
import math

import torch
import torch.nn as nn

from alg_parameters import NetParameters


def _uniform_init(module):
    for param in module.parameters():
        stdv = 1. / math.sqrt(param.size(-1))
        param.data.uniform_(-stdv, stdv)


class AttentionWeight(nn.Module):
    """Single-head attention weights (legacy AttentionWeight without its unused
    w_value / w_out parameters, which never entered the forward pass)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.norm_factor = 1 / math.sqrt(embedding_dim)
        self.w_query = nn.Parameter(torch.Tensor(1, embedding_dim, embedding_dim))
        self.w_key = nn.Parameter(torch.Tensor(1, embedding_dim, embedding_dim))
        _uniform_init(self)

    def forward(self, q, h):
        # q [B, Q, E], h [B, T, E] -> softmax weights [B, Q, T]
        qk = torch.matmul(torch.matmul(q, self.w_query[0]), self.w_key[0].transpose(0, 1))
        return torch.softmax(self.norm_factor * torch.matmul(qk, h.transpose(1, 2)), dim=-1)


class MultiHeadAttention(nn.Module):
    """Legacy MultiHeadAttention (no mask: ALPHA always passed mask=None)."""

    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim // n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)
        self.w_query = nn.Parameter(torch.Tensor(n_heads, embedding_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(n_heads, embedding_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(n_heads, embedding_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(n_heads, self.value_dim, embedding_dim))
        _uniform_init(self)

    def forward(self, q, h):
        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)
        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = q.contiguous().view(-1, input_dim)
        Q = torch.matmul(q_flat, self.w_query).view(self.n_heads, batch_size, n_query, -1)
        K = torch.matmul(h_flat, self.w_key).view(self.n_heads, batch_size, target_size, -1)
        V = torch.matmul(h_flat, self.w_value).view(self.n_heads, batch_size, target_size, -1)
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attention = torch.softmax(U, dim=-1)
        heads = torch.matmul(attention, V)  # [H, B, n_query, value_dim]
        out = torch.mm(heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
                       self.w_out.view(-1, self.embedding_dim))
        return out.view(batch_size, n_query, self.embedding_dim)


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalizer(x.contiguous().view(-1, x.size(-1))).view(*x.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, x, query_index=None):
        """Legacy layer(tgt=x, memory=x). With query_index [B], only the output of token
        x[b, query_index[b]] is computed (returned as [B, 1, E]); every token's output
        depends only on its own query, so this equals the full output at that token."""
        x_norm = self.normalization1(x)  # legacy normalised tgt and memory with normalization1
        if query_index is None:
            h0, q = x, x_norm
        else:
            rows = torch.arange(x.size(0), device=x.device)
            h0, q = x[rows, query_index].unsqueeze(1), x_norm[rows, query_index].unsqueeze(1)
        h = self.multiHeadAttention(q=q, h=x_norm) + h0
        return self.feedForward(self.normalization2(h)) + h


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, n_head) for _ in range(n_layer)])

    def forward(self, nodes, agents, node_query, agent_query):
        """Returns the encoded ego node and ego agent tokens, [B, 1, E] each.
        The same layer processes nodes and agents (legacy)."""
        for layer in self.layers[:-1]:
            nodes = layer(nodes)
            agents = layer(agents)
        last = self.layers[-1]
        return last(nodes, node_query), last(agents, agent_query)


class FocusAttention(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.layer = AttentionWeight(embedding_dim)  # legacy also built an unused Normalization

    def forward(self, ego_node, nodes, agents):
        """ego_node [B*N, 1, E], nodes [B*N, K, E], agents [B, N, E] (identical for every ego)
        -> scaled nodes [B*N, K, E], scaled agents [B*N, N, E]."""
        B, N, E = agents.shape
        nodes = nodes * self.layer(ego_node, nodes).transpose(1, 2)
        w_agents = self.layer(agents, agents)  # [B, ego, N]: row i = ego i's weights
        agents = (w_agents.unsqueeze(-1) * agents.unsqueeze(1)).reshape(B * N, N, E)
        return nodes, agents


class ALPHAGraphEncoder(nn.Module):
    def __init__(self, embedding_dim=NetParameters.EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.static_embedding = nn.Linear(NetParameters.NUM_FEATURE, embedding_dim)
        self.dynamic_embedding = nn.Linear(NetParameters.NUM_INTENTION_FEATURE, embedding_dim)
        self.attention_embedding = FocusAttention(embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.node_agent_embedding = nn.Linear(embedding_dim * 2, NetParameters.NET_SIZE)

    def forward(self, graph_nodes, agent_intent, node_index):
        """graph_nodes [B, N, K, F], agent_intent [B, N, F_int] (one row per agent, shared
        by all egos), node_index [B, N] -> graph feature [B * N, NET_SIZE]."""
        B, N, K, _ = graph_nodes.shape
        rows = torch.arange(B * N, device=graph_nodes.device)
        node_idx = node_index.reshape(B * N).long()
        agent_idx = torch.arange(N, device=graph_nodes.device).repeat(B)  # ego i is agent i

        nodes = self.static_embedding(graph_nodes).reshape(B * N, K, self.embedding_dim)
        agents = self.dynamic_embedding(agent_intent)  # [B, N, E], embedded once
        nodes, agents = self.attention_embedding(nodes[rows, node_idx].unsqueeze(1), nodes, agents)
        ego_node, ego_agent = self.encoder(nodes, agents, node_idx, agent_idx)
        return self.node_agent_embedding(torch.cat((ego_node[:, 0], ego_agent[:, 0]), dim=-1))
