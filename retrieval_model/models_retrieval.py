import json
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torchvision import models
import torch.utils.model_zoo as model_zoo
from gensim.models.keyedvectors import KeyedVectors
import pdb
import torchvision
import math
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, with_attention):
        super(AttentionLayer, self).__init__()
        self.u = torch.nn.Parameter(torch.randn(input_dim)) # u = [2*hid_dim]
        self.u.requires_grad = True
        self.fc = nn.Linear(input_dim, input_dim)
        self.with_attention = with_attention
    def forward(self, x):
        # x = [BS, max_len, 2*hid_dim]
        # a trick used to find the mask for the softmax
        mask = (x!=0)
        mask = mask[:,:,0].bool()
        h = torch.tanh(self.fc(x)) # h = [BS, max_len, 2*hid_dim]
        if self.with_attention == 1: # softmax
            scores = h @ self.u # scores = [BS, max_len], unnormalized importance
        elif self.with_attention == 2: # Transformer
            scores = h @ self.u / math.sqrt(h.shape[-1]) # scores = [BS, max_len], unnormalized importance
        masked_scores = scores.masked_fill(~mask, -1e32)
        alpha = F.softmax(masked_scores, dim=1) # alpha = [BS, max_len], normalized importance

        alpha = alpha.unsqueeze(-1) # alpha = [BS, max_len, 1]
        out = x * alpha # out = [BS, max_len, 2*hid_dim]
        out = out.sum(dim=1) # out = [BS, 2*hid_dim]
        # pdb.set_trace()
        return out, alpha.squeeze(-1)


class IngredientsEncoderRNN(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        hid_dim, 
        z_dim, 
        word2vec_file='data/word2vec_recipes.bin', 
        with_attention=True):
        super(IngredientsEncoderRNN, self).__init__()

        wv = KeyedVectors.load(word2vec_file, mmap='r')
        vec = torch.from_numpy(np.copy(wv.vectors)).float()
        # first two index has special meaning, see load_dict() in utils.py
        emb = nn.Embedding(vec.shape[0]+2, vec.shape[1], padding_idx=0)
        emb.weight.data[2:].copy_(vec)
        # for p in emb.parameters():
        #     p.requires_grad = False
        self.embed_layer = emb
        print('IngredientsEncoderRNN:', emb)
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)

        self.with_attention = with_attention
        if with_attention:
            self.atten_layer = AttentionLayer(2*hid_dim, with_attention)

    def forward(self, sent_list, lengths):
        # sent_list [BS, max_len]
        # lengths [BS]
        x = self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]
        sorted_len, sorted_idx = lengths.sort(0, descending=True) # sorted_idx=[BS], for sorting
        _, original_idx = sorted_idx.sort(0, descending=False) # original_idx=[BS], for unsorting
        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x) # sorted_idx=[BS, max_len, emb_dim]
        sorted_inputs = x.gather(0, index_sorted_idx.long()) # sort by num_words
        packed_seq = rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)
        self.rnn.flatten_parameters()
        # if not self.with_attention:
        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            # pdb.set_trace()
            y, _ = rnn.pad_packed_sequence(
                out, batch_first=True, total_length=20) # y=[BS, max_len, 2*hid_dim], currently in WRONG order!
            unsorted_idx = original_idx.view(-1,1,1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous() # [BS, max_len, 2*hid_dim], now in correct order
            feat, alpha = self.atten_layer(output) # [BS, 2*hid_dim]
            # print('sent', feat.shape) # [BS, 2*hid_dim]
            return feat, alpha
        else:
            _, h = self.rnn(packed_seq) # [2, BS, hid_dim], currently in WRONG order!
            # pdb.set_trace()
            h = h.transpose(0,1) # [BS, 2, hid_dim], still in WRONG order!
            # unsort the output
            unsorted_idx = original_idx.view(-1,1,1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous() # [BS, 2, hid_dim], now in correct order
            feat = output.view(output.size(0), output.size(1)*output.size(2)) # [BS, 2*hid_dim]
            return feat


class IngredientsEncoderFC(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        hid_dim, 
        z_dim, 
        word2vec_file='data/word2vec_recipes.bin', 
        with_attention=True):
        super(IngredientsEncoderFC, self).__init__()

        wv = KeyedVectors.load(word2vec_file, mmap='r')
        vec = torch.from_numpy(wv.vectors).float()
        # first two index has special meaning, see load_dict() in utils.py
        emb = nn.Embedding(vec.shape[0]+2, vec.shape[1], padding_idx=0)
        emb.weight.data[2:].copy_(vec)
        # for p in emb.parameters():
        #     p.requires_grad = False
        self.embed_layer = emb
        print('IngredientsEncoderRNN:', emb)
        
        self.fc1 = nn.Linear(emb_dim, 2*hid_dim)
        self.fc2 = nn.Linear(2*hid_dim, 2*hid_dim)

        self.with_attention = with_attention
        if with_attention:
            self.atten_layer = AttentionLayer(2*hid_dim, with_attention)

    def forward(self, sent_list, lengths):
        # sent_list [BS, max_len]
        # lengths [BS]
        # sent_list [BS, max_len]
        x = self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]
        x = self.fc2(F.relu(self.fc1(x))) # [BS, max_len, 2*hid_dim]
        if not self.with_attention:
            feat = x.sum(dim=1) # [BS, 2*hid_dim]
            return feat
        else:
            feat, alpha = self.atten_layer(x) # [BS, 2*hid_dim]
            # print('ingredients', feat.shape)
            return feat, alpha


class SentenceEncoder(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        hid_dim, 
        z_dim, 
        word2vec_file='data/word2vec_recipes.bin', 
        with_attention=True):
        super(SentenceEncoder, self).__init__()

        wv = KeyedVectors.load(word2vec_file, mmap='r')
        vec = torch.from_numpy(np.copy(wv.vectors)).float()
        # first two index has special meaning, see load_dict() in utils.py
        emb = nn.Embedding(vec.shape[0]+2, vec.shape[1], padding_idx=0)
        emb.weight.data[2:].copy_(vec)
        # for p in emb.parameters():
        #     p.requires_grad = False
        self.embed_layer = emb
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)
        
        self.with_attention = with_attention
        if with_attention:
            self.atten_layer = AttentionLayer(2*hid_dim, with_attention)

    def forward(self, sent_list, lengths):
        # sent_list [BS, max_len]
        # lengths [BS]
        x = self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]
        sorted_len, sorted_idx = lengths.sort(0, descending=True) # sorted_idx=[BS], for sorting
        _, original_idx = sorted_idx.sort(0, descending=False) # original_idx=[BS], for unsorting
        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x) # sorted_idx=[BS, max_len, emb_dim]
        sorted_inputs = x.gather(0, index_sorted_idx.long()) # sort by num_words
        packed_seq = rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)
        self.rnn.flatten_parameters()
        # if not self.with_attention:
        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            # pdb.set_trace()
            y, _ = rnn.pad_packed_sequence(
                out, batch_first=True, total_length=20) # y=[BS, max_len, 2*hid_dim], currently in WRONG order!
            unsorted_idx = original_idx.view(-1,1,1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous() # [BS, max_len, 2*hid_dim], now in correct order
            feat, alpha = self.atten_layer(output) # [BS, 2*hid_dim]
            # print('sent', feat.shape) # [BS, 2*hid_dim]
            return feat, alpha
        else:
            _, h = self.rnn(packed_seq) # [2, BS, hid_dim], currently in WRONG order!
            # pdb.set_trace()
            h = h.transpose(0,1) # [BS, 2, hid_dim], still in WRONG order!
            # unsort the output
            unsorted_idx = original_idx.view(-1,1,1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous() # [BS, 2, hid_dim], now in correct order
            feat = output.view(output.size(0), output.size(1)*output.size(2)) # [BS, 2*hid_dim]
            return feat


class DocEncoder(nn.Module):
    def __init__(self, sent_encoder, hid_dim, with_attention):
        super(DocEncoder, self).__init__()
        self.sent_encoder = sent_encoder
        self.rnn = nn.GRU(
            input_size=2*hid_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)
        if with_attention:
            self.atten_layer_sent = AttentionLayer(2*hid_dim, with_attention)
        self.with_attention = with_attention
    
    def forward(self, doc_list, n_insts, n_words_each_inst):
        # doc_list=[BS, max_len, max_len]
        # n_insts = [BS]
        # n_words_each_inst = [BS, 20]
        embs = []
        attentions_words_each_inst =[]
        for i in range(len(n_insts)):
            doc = doc_list[i]
            ln = n_insts[i] # how many steps
            sent_lns = n_words_each_inst[i, :n_words_each_inst[i].nonzero(as_tuple=False).shape[0]] # len of each step

            if self.with_attention:
                emb_doc, alpha = self.sent_encoder(doc[:ln], sent_lns) # [?, 2*hid_dim]
                attentions_words_each_inst.append(alpha) # e.g. if lns=[5,1,14,7, ...], then attentions_words_each_inst=[[5,20], [1,20], [14,20], [7,20], ...] with length=BS
            else:
                emb_doc = self.sent_encoder(doc[:ln], sent_lns) # [?, 2*hid_dim]
            embs.append(emb_doc)
        
        embs = sorted(embs, key=lambda x: -x.shape[0]) # [BS, [?, 2*hid_dim]]
        packed_seq = rnn.pack_sequence(embs)
        _, sorted_idx = n_insts.sort(0, descending=True) # sorted_idx=[BS], for sorting
        _, original_idx = sorted_idx.sort(0, descending=False) # original_idx=[BS], for unsorting

        self.rnn.flatten_parameters()
        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            y, _ = rnn.pad_packed_sequence(
                out, batch_first=True, total_length=20) # y=[BS, max_len, 2*hid_dim], currently in WRONG order!
            # pdb.set_trace()
            unsorted_idx = original_idx.view(-1,1,1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous() # [BS, max_len, 2*hid_dim], now in correct order
            out, attentions_each_inst = self.atten_layer_sent(output)
            # print('instructions', feat.shape)
            return out, attentions_each_inst, attentions_words_each_inst
        else:
            _, h = self.rnn(packed_seq) # [2, BS, hid_dim], currently in WRONG order!
            h = h.transpose(0,1) # [BS, 2, hid_dim], still in WRONG order!
            # unsort the output
            unsorted_idx = original_idx.view(-1,1,1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous() # [BS, 2, hid_dim], now in correct order
            feat = output.view(output.size(0), output.size(1)*output.size(2)) # [BS, 2*hid_dim]
            # print('instructions', feat.shape)
            return feat


class TextEncoder(nn.Module):
    def __init__(
        self, 
        emb_dim, hid_dim, z_dim, 
        word2vec_file, 
        with_attention=0,
        text_info='010', 
        ingrs_enc_type='rnn'
        ):
        super(TextEncoder, self).__init__()
        if ingrs_enc_type == 'rnn':
            self.ingrs_encoder = IngredientsEncoderRNN(
                emb_dim, hid_dim, z_dim,
                word2vec_file=word2vec_file,
                with_attention=with_attention)
        elif ingrs_enc_type == 'fc':
            self.ingrs_encoder = IngredientsEncoderFC(
                emb_dim, hid_dim, z_dim,
                word2vec_file=word2vec_file,
                with_attention=with_attention)
        
        self.sent_encoder = SentenceEncoder(
            emb_dim=emb_dim, hid_dim=hid_dim, z_dim=z_dim, word2vec_file=word2vec_file, 
            with_attention=with_attention)
        self.doc_encoder = DocEncoder(
            self.sent_encoder, 
            hid_dim, 
            with_attention
        )
        self.with_attention = with_attention
        self.text_info = text_info
        num_ones = text_info.count('1')
        self.bn = nn.BatchNorm1d(2*num_ones*hid_dim)
        self.fc = nn.Linear(2*num_ones*hid_dim, z_dim)

    def forward(self, title, title_len, ingredients, n_ingrs, instructions, n_insts, insts_lens):
        if self.with_attention:
            feat_title, alpha_title = self.sent_encoder(title, title_len)
            feat_ingredients, alpha_ingredients = self.ingrs_encoder(ingredients, n_ingrs)
            feat_instructions, alpha_instructions, alpha_words = self.doc_encoder(instructions, n_insts, insts_lens)
        else:
            feat_title = self.sent_encoder(title, title_len)
            feat_ingredients = self.ingrs_encoder(ingredients, n_ingrs)
            feat_instructions = self.doc_encoder(instructions, n_insts, insts_lens)
        
        if self.text_info == '100':
            feat = feat_title
            attentions = [alpha_title, None, None, None] if self.with_attention else None
        elif self.text_info == '010':
            feat = feat_ingredients
            attentions = [None, alpha_ingredients, None, None] if self.with_attention else None
        elif self.text_info == '001':
            feat = feat_instructions
            attentions = [None, None, alpha_instructions, alpha_words] if self.with_attention else None
        elif self.text_info == '111':
            feat = torch.cat([feat_title, feat_ingredients, feat_instructions], dim=1)
            attentions = [alpha_title, alpha_ingredients, alpha_instructions, alpha_words] if self.with_attention else None
        
        feat = self.fc(self.bn(feat))
        feat = F.normalize(feat, p=2, dim=1)
        return feat, attentions


# Image Encoder
def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class Resnet(nn.Module):
    def __init__(self, ckpt_path=None):
        super(Resnet, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        num_feat = resnet.fc.in_features
        resnet.fc = nn.Linear(num_feat, 101)
        if ckpt_path:
            resnet.load_state_dict(clean_state_dict(torch.load(ckpt_path)))
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, image_list):
        BS = image_list.shape[0]
        return self.encoder(image_list).view(BS, -1)

class ImageEncoder(nn.Module):
    def __init__(self, z_dim, ckpt_path=None):
        super(ImageEncoder, self).__init__()
        self.resnet = Resnet(ckpt_path)
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, z_dim),
            nn.Tanh()
        )
    
    def forward(self, image_list):
        feat = self.resnet(image_list)
        feat = self.bottleneck(feat)
        # print('image', feat.shape)
        return F.normalize(feat, p=2, dim=1)