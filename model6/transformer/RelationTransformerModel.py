# Follow the implementation in https://github.com/yahoo/object_relation_transformer/blob/master/models/RelationTransformerModel.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from utils import box_utils as utils
#utils.PositionalEmbedding()
import copy
import math
import numpy as np

from .CaptionModel import CaptionModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # lambda x:x
        self.tgt_embed = tgt_embed  # Embeddings + PositionalEncoding
        self.generator = generator


    def forward(self, src, boxes, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, boxes, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, boxes, src_mask):
        # memory(encoder output): (64, 8, 512) --> (batch_size, max_len, d_model)
        return self.encoder(self.src_embed(src), boxes, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        out = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return out

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1), self.proj(x)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        # layer=EncoderLayer, N=6
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 复制６份
        self.norm = LayerNorm(layer.size)

    def forward(self, x, box, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, box, mask)
            
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # region attention
        #self.ge_attn = ge_attn  # box relation
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, box, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, box, mask))
        #x = self.sublayer[1](x, lambda x: self.ge_attn(x, x, x, box, node_num, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        # x:(64,16,512)
        m = memory  # (64, 8, 512)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # (64, 16, 512)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # (64, 16, 512)
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    scores = scores.float()
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # 1. mask-multi-head-attention
        # query is the caption feature: (64, 512)
        # k and v is the seq input: (64, 16, 512)
        # mask: (64, 16, 16)

        # 2. multi-head-attention
        # query is the mask-multi-head-attention output: (64, 16, 512)
        # k and v is the encoder output(memory): (64, 8, 512)
        # mask: (512, 16, 16)

        if query.dim() == 2:
            q_size = query.size()
            num = key.size(1)
            query = query.repeat(1, 1, num).view(q_size[0], num, q_size[-1])

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # (64, 1, 16, 16)
        nbatches = query.size(0)

        #print(query.shape, key.shape, value.shape)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # q,k,v: (64, 8, 16, 64)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    #attention weights
    scaled_dot = torch.matmul(w_q,w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)  # (512, 33, 33)
    scaled_dot = scaled_dot.float()

    #w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix
    w_a = scaled_dot
    #w_a = scaled_dot.view(N,N)

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a

    if mask is not None:    # mask: (64, 1, 1, 8)
        w_mn = w_mn.masked_fill(mask == 0, -1e9)
    
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn,w_v)

    return output, w_mn

class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0  # 512/8=64
        self.trignometric_embedding=trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        # matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True),8)
        #self.linears2 = nn.Linear(64, 512)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        # q: torch.Size([64, 512])
        # k=v: torch.Size([64, 8, 512])
        # input_box: torch.Size([64, 8, 4])
        # node_num: list, len = 64
        # mask: torch.Size([64, 1, 8])
        #print(input_query.shape, input_key.shape, input_value.shape)
        if input_query.dim() == 2:
            q_size = input_query.size()
            num = input_key.size(1)
            input_query = input_query.repeat(1, 1, num).view(q_size[0], num, q_size[-1])

        #assert input_query.size() == input_key.size() and input_query.size() == input_value.size()

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # torch.Size([64, 1, 1, 8])
        nbatches = input_query.size(0)  # 64

        # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = utils.BoxRelationalEmbedding(input_box, trignometric_embedding=self.trignometric_embedding)
        # relative_geometry_embeddings: torch.Size([64, 8, 8, 64])
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1,self.dim_g)  # (64*8*8, 64)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]
        # q,k,v: torch.Size([64, 8, 8, 64]) --> (batch_size, max_node_num, 8head, d_k)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3]) # (64, 8, 8)
        box_size_per_head.insert(1, 1)  # [64, 1, 8, 8]
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head),1)  # (64*8*8, 8)
        relative_geometry_weights = F.relu(relative_geometry_weights)  # W_G(mn)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        # An extra internal skip connection is added. This is only
        # kept here for compatibility with some legacy models. In
        # general, there is no advantage in using it, as there is
        # already an outer skip connection surrounding this layer.
        #x = self.linears2(x)
        if self.legacy_extra_skip:
            x = input_value + x
        
        return self.linears[-1](x)  # (64, 8, 512) --> (batch_size, src_max_len, d_model)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationTransformerModel(CaptionModel):

    def make_model(self, src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1,
                   trignometric_embedding=True, legacy_extra_skip=False):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        bbox_attn = BoxMultiHeadedAttention(h, d_model, trignometric_embedding, legacy_extra_skip)
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        #position = BoxEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(bbox_attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, vocab_size, pad_idx=0, start_idx=1, end_idx=2):
        super(RelationTransformerModel, self).__init__()

        #self.opt = opt
        self.vocab_size = vocab_size
        self.input_encoding_size = 512

        self.drop_prob_lm = 0.4
        self.seq_length = 16

        self.sg_feat_size = 1024
        self.region_feat_size = 2048
        # 0,1,2
        self.special_idx = {
            '<pad>':pad_idx,
            '<bos>':start_idx,
            '<eos>':end_idx
        }

        self.use_bn = 0

        self.ss_prob = 0.0  # Schedule sampling probability

        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.sg_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.sg_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn==2 else ())))

        self.att_embed2 = nn.Sequential(*(
                ((nn.BatchNorm1d(self.region_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.region_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))


        self.box_trignometric_embedding = True
        self.legacy_extra_skip = False

        # tgt_vocab = self.vocab_size + 1  # +1 ???????
        tgt_vocab = self.vocab_size
        self.model = self.make_model(
            src_vocab=0, tgt_vocab=tgt_vocab, N=6, d_model=512, d_ff=2048,
            trignometric_embedding=self.box_trignometric_embedding,
            legacy_extra_skip=self.legacy_extra_skip)

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, att_feats, cap_feats, node_num, att_masks=None, boxes=None, seq=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)  # 1024 --> 512
        boxes = self.clip_att(boxes, att_masks)[0]

        #cap_feats = pack_wrapper(self.att_embed2, cap_feats, att_masks)  # 2048 --> 512

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)  # torch.Size([64, 8])
            #att_masks = (att_feats.data > 0)
            for i, j in enumerate(node_num):
                att_masks[i, j:] = 0
        att_masks = att_masks.unsqueeze(-2)  # torch.Size([64, 1, 8])

        if seq is not None:
            # crop the last one  # <'eos'>
            seq = seq[:,:-1]  #
            seq_mask = (seq.data > 0)
            seq_mask[:,0] = 1   # torch.Size([64, 16])

            seq_mask = seq_mask.unsqueeze(-2)  # torch.Size([64, 1, 16])
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)  # torch.Size([64, 16, 16])
        else:
            seq_mask = None

        return att_feats, boxes, att_masks, seq, seq_mask

    def _forward(self, region_feat, region_box, caps, caps_len, sg_feat, sg_box, sg_node_num, att_masks=None):
        """

        Args:
            region_feat: torch.Size([64, 2048])
            region_box: np.array([64, 4])
            caps: torch.Size([64, 17])
            caps_len: np.array([64,])
            sg_feat: torch.Size([64, 8, 1024])
            sg_box: np.array([64, 8, 4])
            sg_node_num: list (64)
            att_masks: None

        Returns:

        """
        
        node_feats, node_boxes, att_masks, seq, seq_mask = self._prepare_feature(sg_feat, region_feat, sg_node_num, att_masks, sg_box, caps)
        
        out = self.model(src=node_feats, boxes=node_boxes, tgt=seq.long(), src_mask=att_masks, tgt_mask=seq_mask)
        # out: (batch_size, tgt_max_len, d_model) --> (64, 16, 512)
        outputs = self.model.generator(out)[1]  # (64, 16, 10629)
        # outputs : Linear + log_softmax
        # outputs2: Linear
        return outputs


    def get_logprobs_state(self, it, memory, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)  # torch.Size([1000, 1])
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        
        tgt_mask = subsequent_mask(ys.size(1)).to(memory.device)
        
        out = self.model.decode(memory, mask, ys, tgt_mask)
        
        logprobs = self.model.generator(out[:, -1])[0]  # linear + log_softmax
        # logprobs: (200, 10629)
        # ys.unsqueeze(0) --> torch.Size([1, 200, 1])
        return logprobs, [ys.unsqueeze(0)]

    def _sample_beam(self, att_feats, boxes, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes)
        memory = self.model.encode(att_feats, boxes, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            tmp_memory = memory[k:k+1].expand(*((beam_size,)+memory.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = att_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_memory, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample_(self, fc_feats, att_feats, boxes, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        if sample_max:
            with torch.no_grad():
                seq_, seqLogprobs_ = self._sample_(fc_feats, att_feats, boxes, att_masks, opt)

        batch_size = att_feats.shape[0]

        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes)
        memory = self.model.encode(att_feats, boxes, att_masks)
        ys = torch.zeros((batch_size, 1), dtype=torch.long).to(att_feats.device)

        seq = att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)

        for i in range(self.seq_length):
            out = self.model.decode(memory, att_masks,
                               ys,
                               subsequent_mask(ys.size(1))
                                        .to(att_feats.device))
            logprob = self.model.generator(out[:, -1])
            if sample_max:
                sampleLogprobs, next_word = torch.max(logprob, dim = 1)
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprob.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprob.data, temperature))
                next_word = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, next_word) # gather the logprobs at sampled positions

            seq[:,i] = next_word
            seqLogprobs[:,i] = sampleLogprobs
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        assert (seq*((seq_>0).long())==seq_).all(), 'seq doens\'t match'
        assert (seqLogprobs*((seq_>0).float()) - seqLogprobs_*((seq_>0).float())).abs().max() < 1e-5, 'logprobs doens\'t match'
        return seq, seqLogprobs

    def _sample(self, region_feat, region_box, sg_feat, sg_box, sg_node_num, att_masks=None, opt={}):
        
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        device = region_feat.device
        
        if beam_size > 1:
            return self._sample_beam(att_masks, opt)

        batch_size = region_feat.shape[0]  # dataset batchsize=1 --> batch_size=300
        node_feats, node_boxes, att_masks, seq, seq_mask = self._prepare_feature(sg_feat, region_feat, sg_node_num, att_masks, sg_box)
        assert seq == None and seq_mask == None
        state = None
        
        memory = self.model.encode(node_feats, node_boxes, att_masks)  # (300, 20, 512)

        seq = node_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)  # (300, 16)
        seqLogprobs = node_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length):
            if t == 0:  # input <bos>
                #it = fc_feats.new_zeros(batch_size, dtype=torch.long)
                it = torch.ones(batch_size, dtype=torch.long).to(device) * self.special_idx['<bos>']

            logprobs, state = self.get_logprobs_state(it, memory, att_masks, state)
            # (200, 10629) \ [tensor0(1, 200, 1)]
            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)  # (200), (200)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                #unfinished = it > 0  # pad == 0
                unfinished = it != self.special_idx['<eos>']
            else:
                #unfinished = unfinished * (it > 0)
                unfinished = unfinished * (it != self.special_idx['<eos>'])
            it = it * unfinished.type_as(it)  # unfinished.type_as(it) --> bool to long(true/false to 0/1)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def nopeak_mask(self, size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        np_mask = Variable(torch.from_numpy(np_mask) == 0)
        np_mask = np_mask.cuda()
        return np_mask

    def init_vars(self, node_feats, node_boxes, att_masks, node_num_batch, rel_feat_batch, box_features, beam_size):
    
        init_tok = self.special_idx['<sos>']
        
        e_output = self.model.encode(node_feats, node_boxes, att_masks, node_num_batch, rel_feat_batch, box_features)
    
        outputs = torch.LongTensor([[init_tok]])
        
        outputs = outputs.cuda()
    
        trg_mask = self.nopeak_mask(1)
    
        out = self.model.out(self.model.decoder(outputs, e_output, trg_mask))
        out = F.softmax(out, dim=-1)
    
        probs, ix = out[:, -1].data.topk(beam_size)
        log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
        outputs = torch.zeros(beam_size, self.max_len).long()
        
        outputs = outputs.cuda()
        outputs[:, 0] = init_tok
        outputs[:, 1] = ix[0]
    
        e_outputs = torch.zeros(beam_size, e_output.size(-2),e_output.size(-1))
        
        e_outputs = e_outputs.cuda()
        e_outputs[:, :] = e_output[0]
    
        return outputs, e_outputs, log_scores

    def k_best_outputs(self, outputs, out, log_scores, i, k):
    
        probs, ix = out[:, -1].data.topk(k)
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
        k_probs, k_ix = log_probs.view(-1).topk(k)
    
        row = k_ix // k
        col = k_ix % k

        outputs[:, :i] = outputs[row, :i]
        outputs[:, i] = ix[row, col]

        log_scores = k_probs.unsqueeze(0)
    
        return outputs, log_scores

    def _beam_search2(self, box_features, caption_gt, caption_length, node_box_batch, node_feat_batch, rel_feat_batch, node_num_batch, att_masks=None, opt={}):
        print('test model')
        assert caption_gt == None and caption_length == None
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 3)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        device = box_features.device

        node_feats, box_features, node_boxes, seq, att_masks, seq_mask = self._prepare_feature(node_feat_batch, box_features, node_num_batch, att_masks, node_box_batch)
        assert seq == None and seq_mask == None

        outputs, e_outputs, log_scores = self.init_vars(node_feats, node_boxes, att_masks, node_num_batch, rel_feat_batch, box_features, beam_size)
        eos_tok = self.special_idx['<eos>']
        
        ind = None
        for i in range(2, opt.max_len):
    
            trg_mask = self.nopeak_mask(i, opt)

            out = self.model.out(self.model.decoder(outputs[:,:i], e_outputs, trg_mask))

            out = F.softmax(out, dim=-1)
    
            outputs, log_scores = self.k_best_outputs(outputs, out, log_scores, i, beam_size)
        
            ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
            sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
            for vec in ones:
                i = vec[0]
                if sentence_lengths[i]==0: # First end symbol has not been found yet
                    sentence_lengths[i] = vec[1] # Position of first end symbol

            num_finished_sentences = len([s for s in sentence_lengths if s > 0])

            if num_finished_sentences == opt.k:
                alpha = 0.7
                div = 1/(sentence_lengths.type_as(log_scores)**alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break
    
        if ind is None:
            length = (outputs[0]==eos_tok).nonzero()[0]
            #return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
            return outputs[0][1:length]
    
        else:
            length = (outputs[ind]==eos_tok).nonzero()[0]
            #return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
            return outputs[ind][1:length]
