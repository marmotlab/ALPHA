import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
import math

from alg_parameters import *


class AttentionWeight(nn.Module):
    def __init__(self, embedding_dim, n_heads=1):
        super(AttentionWeight, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U[mask.bool()] = -np.inf
            U[mask] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            # attnc[mask.bool()] = 0
            attnc[mask] = 0
            attention = attnc
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U[mask.bool()] = -np.inf
            U[mask] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            # attnc[mask.bool()] = 0
            attnc[mask] = 0
            attention = attnc
        # print(attention)

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        # print(f"memory is {memory.shape}")
        memory = self.normalization1(memory)
        # print(f"memory 2 is {memory.shape}")
        # print(f"tgt is {tgt.shape}")
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class Encoder(nn.Module):
    # how many layers of encoder
    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, all_nodes_embedding, all_agents_embedding, mask):
        for layer in self.layers:
            all_nodes_embedding = layer(tgt=all_nodes_embedding, memory=all_nodes_embedding, mask=mask)
            all_agents_embedding = layer(tgt=all_agents_embedding, memory=all_agents_embedding, mask=mask)
        return all_nodes_embedding, all_agents_embedding

class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.contiguous().view(-1, input.size(-1))).view(*input.size())


def normalized_columns_initializer(weights, std=1.0):
    """weight initializer"""
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    """initialize weights"""
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FocusAttention(nn.Module):
    def __init__(self, embedding_dim=128, n_head=1):
        super(FocusAttention, self).__init__()
        self.layer = AttentionWeight(embedding_dim)
        self.normalization = Normalization(embedding_dim)
    def forward(self, current_node_embedding, all_nodes_embedding, current_agent_embedding, all_agents_embedding, mask):
        node_attention_weight = self.layer(current_node_embedding, all_nodes_embedding, mask=mask)
        # print(f"number of nodes are {all_nodes_embedding.size(1)}")
        node_attention_weight = node_attention_weight.reshape(-1, all_nodes_embedding.size(1))

        node_temp_1 = node_attention_weight.repeat_interleave(all_nodes_embedding.size(2), dim=1)
        node_temp_2 = node_temp_1.reshape(all_nodes_embedding.size(0), all_nodes_embedding.size(1), all_nodes_embedding.size(2))
        new_all_nodes_embedding = torch.mul(all_nodes_embedding, node_temp_2)
        # zoom feature scale
        # new_all_nodes_embedding = new_all_nodes_embedding * NetParameters.NUM_NODES

        agent_attention_weight = self.layer(current_agent_embedding, all_agents_embedding, mask=mask)
        agent_attention_weight = agent_attention_weight.reshape(-1, all_agents_embedding.size(1))

        agent_temp_1 = agent_attention_weight.repeat_interleave(all_agents_embedding.size(2), dim=1)
        agent_temp_2 = agent_temp_1.reshape(all_agents_embedding.size(0), all_agents_embedding.size(1), all_agents_embedding.size(2))
        new_all_agents_embedding = torch.mul(all_agents_embedding, agent_temp_2)

        return new_all_nodes_embedding, new_all_agents_embedding


class ALPHANet(nn.Module):
    """network with transformer-based communication mechanism"""

    def __init__(self, embedding_dim):
        """initialization"""
        super(ALPHANet, self).__init__()
        # observation encoder
        self.conv1 = nn.Conv2d(NetParameters.NUM_CHANNEL, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.conv1a = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.conv1b = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2a = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2b = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE - NetParameters.GOAL_REPR_SIZE, 3,
                               1, 0)
        self.fully_connected_1 = nn.Linear(NetParameters.VECTOR_LEN, NetParameters.GOAL_REPR_SIZE)
        self.fully_connected_2 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        self.fully_connected_3 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)

        self.lstm_memory = nn.LSTMCell(input_size=NetParameters.NET_SIZE, hidden_size=NetParameters.NET_SIZE)

        # output heads
        self.policy_layer = nn.Linear(NetParameters.NET_SIZE, EnvParameters.N_ACTIONS)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.sigmoid_layer = nn.Sigmoid()
        self.value_layer = nn.Linear(NetParameters.NET_SIZE, 1)
        self.blocking_layer = nn.Linear(NetParameters.NET_SIZE, 1)
        self.apply(weights_init)

        # node and agent encoder
        self.embedding_dim = embedding_dim
        # self.node_embedding = nn.Linear(NetParameters.NUM_FEATURE, embedding_dim)
        self.agent_embedding = nn.Linear(2, embedding_dim)
        self.goals_embedding = nn.Linear(2, embedding_dim)
        self.agent_goals_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.node_agent_embedding = nn.Linear(embedding_dim * 2, NetParameters.NET_SIZE)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.attention_embedding = FocusAttention(embedding_dim=embedding_dim, n_head=1)
        self.fully_connected_4 = nn.Linear(NetParameters.NET_SIZE * 2, NetParameters.NET_SIZE)

        # cooperation input part
        self.static_embedding = nn.Linear(NetParameters.NUM_FEATURE, embedding_dim)
        self.dynamic_embedding = nn.Linear(NetParameters.NUM_INTENTION_FEATURE, embedding_dim)
        # self.combine_nodes_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

    @autocast()
    def forward(self, obs, vector, graph_nodes, agent_intent, current_node_index,
                current_agent_index, input_state):
        # print(f"current_node_index is {current_node_index.shape}")
        # adjust the dimension of input
        # print(f"graph nodes is {graph_nodes[:, 0, 0, :]}")
        static_feature = graph_nodes
        dynamic_feature = agent_intent
        # print(f"static feature is {static_feature[:, 0, 0, :]}")
        # print(f"dynamic feature is {dynamic_feature[:, 0, 0, :]}")
        static_embedding = self.static_embedding(static_feature)
        agent_embedding = self.dynamic_embedding(dynamic_feature)
        num_agent = obs.shape[1]
        obs = torch.reshape(obs, (-1, 4, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))
        vector = torch.reshape(vector, (-1, NetParameters.VECTOR_LEN))

        node_embedding = static_embedding
        # print(f"node embedding is {node_embedding.shape}")
        # agent_embedding = self.agent_embedding(graph_agents)
        # goals_embedding = self.goals_embedding(graph_goals)
        # agent_embedding = self.agent_goals_embedding(torch.cat((agent_embedding, goals_embedding), dim=-1))

        graph_feature = torch.tensor([]).to(obs.device)
        for i in range(num_agent):
            current_all_node_embedding = node_embedding[:, i, :, :]  # (batch, num_nodes, embedding_dim)
            current_all_agent_embedding = agent_embedding[:, i, :, :]
            node_index = current_node_index[:, i, :].unsqueeze(1)
            agent_index = current_agent_index[:, i, :].unsqueeze(1)

            current_node_embedding = torch.gather(current_all_node_embedding, 1,
                                                  node_index.repeat(1, 1, self.embedding_dim))  # (batch, 1, embedding_dim)

            current_agent_embedding = torch.gather(current_all_agent_embedding, 1,
                                                   agent_index.repeat(1, 1,
                                                                      self.embedding_dim))

            # calculate the attention weight
            for _ in range(1):
                current_all_node_embedding, current_all_agent_embedding = self.attention_embedding(
                    current_node_embedding, current_all_node_embedding,
                    current_agent_embedding,
                    current_all_agent_embedding,
                    mask=None)

            # print(f"current agent embedding is {current_agent_embedding.shape}")
            current_all_node_embedding, current_all_agent_embedding = self.encoder(
                current_all_node_embedding, current_all_agent_embedding, mask=None)
            current_node_feature = torch.gather(current_all_node_embedding, 1,
                                                node_index.repeat(1, 1, self.embedding_dim))
            current_agent_feature = torch.gather(current_all_agent_embedding, 1,
                                                 agent_index.repeat(1, 1, self.embedding_dim))
            current_node_feature_final = self.node_agent_embedding(
                torch.cat((current_node_feature, current_agent_feature), dim=-1))
            graph_feature = torch.cat((graph_feature, current_node_feature_final), dim=1)  # TODO: DIM=0? or 1
            # print(f"graphe feature is {graph_feature.shape}")

        # current_node_feature = (batch, 1, embedding_dim)
        # current_agent_feature = (batch, 1, embedding_dim)
        graph_feature = torch.reshape(graph_feature, (-1, self.embedding_dim))
        # print(f"graph feature is {graph_feature.shape}")
        # matrix input
        x_1 = F.relu(self.conv1(obs))
        x_1 = F.relu(self.conv1a(x_1))
        x_1 = F.relu(self.conv1b(x_1))
        x_1 = self.pool1(x_1)
        x_1 = F.relu(self.conv2(x_1))
        x_1 = F.relu(self.conv2a(x_1))
        x_1 = F.relu(self.conv2b(x_1))
        x_1 = self.pool2(x_1)
        x_1 = self.conv3(x_1)
        x_1 = F.relu(x_1.view(x_1.size(0), -1))
        # vector input
        x_2 = F.relu(self.fully_connected_1(vector))
        # Concatenation
        x_3 = torch.cat((x_1, x_2), -1)
        # print(f"x3 is {x_3.shape}")
        # concatenation all
        x_3 = torch.cat((x_3, graph_feature), -1)
        x_3 = self.fully_connected_4(x_3)

        h1 = F.relu(self.fully_connected_2(x_3))
        h1 = self.fully_connected_3(h1)
        h2 = F.relu(h1 + x_3)
        # LSTM cell
        memories, memory_c = self.lstm_memory(h2, input_state)
        output_state = (memories, memory_c)
        memories = torch.reshape(memories, (-1, num_agent, NetParameters.NET_SIZE))

        policy_layer = self.policy_layer(memories)
        policy = self.softmax_layer(policy_layer)
        policy_sig = self.sigmoid_layer(policy_layer)
        value = self.value_layer(memories)
        blocking = torch.sigmoid(self.blocking_layer(memories))
        return policy, value, blocking, policy_sig, output_state, policy_layer
