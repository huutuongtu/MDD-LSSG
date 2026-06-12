import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class LookUpGCN(nn.Module):
    """
    Linguistic encoder:
    - Nodes = phonemes
    - Aggregation from incoming neighbors (j -> i)
    """
    def __init__(self, num_phonemes, embed_dim, hidden_channels, out_channels, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(num_phonemes, embed_dim, padding_idx=pad_id)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.5)

        if pad_id is not None:
            with torch.no_grad():
                self.embedding.weight[pad_id].zero_()

        self.conv1 = GCNConv(embed_dim, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_ids, edge_index, edge_weight):
        x = self.embedding(node_ids)

        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.norm1(x + self.dropout(x1))

        x2 = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.norm2(x + self.dropout(x2))

        return x


class GCN_MDD(Wav2Vec2PreTrainedModel):
    """
    Acoustic–Linguistic Phoneme ASR with CTC
    """
    def __init__(self, config, vocab_size: int, pad_id: int):
        super().__init__(config)

        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # Acoustic encoder
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )

        # Linguistic encoder (graph)
        self.look_up_model = LookUpGCN(
            num_phonemes=vocab_size,
            embed_dim=config.hidden_size,
            hidden_channels=config.hidden_size,
            out_channels=config.hidden_size,
            pad_id=pad_id,
        )

        # Cross-attention: audio queries linguistic
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=16,
            dropout=0.2,
            batch_first=True,
        )

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)

        # Final classifier 
        self.classifier = nn.Linear(
            config.hidden_size * 2, vocab_size
        )

        # Graph buffers
        self.register_buffer(
            "indices",
            torch.arange(self.vocab_size, dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "edge_weight",
            torch.empty((0,), dtype=torch.float),
            persistent=False
        )

        self.post_init()

    def set_graph(self, edge_index, edge_weight):
        self.edge_index = edge_index.contiguous()
        self.edge_weight = edge_weight.contiguous()

    def look_up_table(self, canonical_ids):
        """
        Returns linguistic embeddings for canonical phonemes
        """
        all_nodes = self.look_up_model(
            self.indices, self.edge_index, self.edge_weight
        )
        return all_nodes[canonical_ids]   # (B, N_can, H)

    def get_look_up_table(self, ):
        all_nodes = self.look_up_model(self.indices, self.edge_index, self.edge_weight)
        return all_nodes

    def forward(self, audio_input, canonical):
        """
        Returns:
            logits: (B, T_audio, vocab_size)
        """

        # Acoustic encoder
        acoustic = self.wav2vec2(audio_input).last_hidden_state   # (B, T_audio, H)

        # Linguistic encoder (graph)
        linguistic = self.look_up_table(canonical)  # (B, T_can, H)

        # Cross-attention (audio queries linguistic)
        Q = self.linear_q(acoustic)
        K = self.linear_k(linguistic)
        V = self.linear_v(linguistic)

        context, _ = self.cross_attn(
            Q, K, V,
            key_padding_mask=canonical.eq(self.pad_id)
        )  # (B, T_audio, H)

        # Concatenate acoustic + linguistic context
        fused = torch.cat([acoustic, context], dim=-1)

        # Phoneme logits (CTC-ready)
        logits = self.classifier(fused)  # (B, T_audio, vocab)

        return logits
