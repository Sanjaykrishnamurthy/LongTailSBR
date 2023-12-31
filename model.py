import numpy as np
import torch
import torch.nn as nn   

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork, self).__init__()
        self.gate_fc1 = nn.Linear(input_dim, input_dim)
        self.gate_fc2 = nn.Linear(input_dim, input_dim)
    
    def forward(self, tensor1, tensor2):
        batch_size, seq_len, feature_dim = tensor1.size()
        tensor1_flat = tensor1.reshape(-1, feature_dim)
        tensor2_flat = tensor2.reshape(-1, feature_dim)
        
        gate = torch.sigmoid(self.gate_fc1(tensor1_flat) + self.gate_fc2(tensor2_flat))
        gate = gate.view(batch_size, seq_len, feature_dim)
        
        # Combine the tensors using the gate
        combined_tensor = gate * tensor1 + (1 - gate) * tensor2
        return combined_tensor
    
class CLTSBR(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(CLTSBR, self).__init__()
        self.sess_emb_dict = torch.randn(user_num, args.hidden_units, device=args.device)
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.gate = GatingNetwork(args.hidden_units)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden_units, nhead=2)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            

    def log2feats(self, log_seqs, seq_pop):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seq_pop = torch.FloatTensor(seq_pop).to(self.dev)
        seq_pop = seq_pop * torch.ones_like(seqs)
        seqs += seq_pop
        
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) 

        tl = seqs.shape[1] 
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) 

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, n_sess, seq_pop):   
        log_feats = self.log2feats(log_seqs, seq_pop) 
        self.sess_emb_dict[user_ids] = log_feats.sum(dim=1) 
        top_neighbor_feats = self.sess_emb_dict[n_sess] 
        log_feats = log_feats.transpose(0, 1) 
        top_neighbor_feats = top_neighbor_feats.transpose(0, 1)  
        top_neighbor_feats = self.decoder_layer(log_feats, top_neighbor_feats)
        
        log_feats = log_feats.transpose(0, 1) 
        top_neighbor_feats = top_neighbor_feats.transpose(0, 1)  
        pos_embs_pred = self.gate(log_feats, top_neighbor_feats)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (pos_embs_pred * pos_embs).sum(dim=-1)
        neg_logits = (pos_embs_pred * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs) 
        self.sess_emb_dict[user_ids] = log_feats
        
        final_feat = log_feats[:, -1, :] 
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) 
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits 
