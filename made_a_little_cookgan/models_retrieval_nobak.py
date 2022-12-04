import json
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torchvision import models
from gensim.models.keyedvectors import KeyedVectors
import pdb


def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.u = torch.nn.Parameter(torch.randn(input_dim)) # u = [2*hid_dim]
        self.u.requires_grad = True
        self.fc = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        # x = [BS, num_vec, 2*hid_dim]
        mask = (x!=0)
        # a trick used to find the mask for the softmax
        mask = mask[:,:,0].bool()
        h = torch.tanh(self.fc(x)) # h = [BS, num_vec, 2*hid_dim]
        tmp = h @ self.u # tmp = [BS, num_vec], unnormalized importance
        masked_tmp = tmp.masked_fill(~mask, -1e32)
        alpha = F.softmax(masked_tmp, dim=1) # alpha = [BS, num_vec], normalized importance
        alpha = alpha.unsqueeze(-1) # alpha = [BS, num_vec, 1]
        out = x * alpha # out = [BS, num_vec, 2*hid_dim]
        out = out.sum(dim=1) # out = [BS, 2*hid_dim]
        # pdb.set_trace()
        return out


class InstEmbedLayer(nn.Module):
    def __init__(self, data_dir, emb_dim):
        super(InstEmbedLayer, self).__init__()
        self.data_dir = data_dir
        path = os.path.join(self.data_dir, 'word2vec.bin')
        # model = KeyedVectors.load_word2vec_format(path, binary=True)
        wv = KeyedVectors.load(path, mmap='r')
        vec = torch.from_numpy(wv.vectors).float()
        # first three index has special meaning, see utils.py
        emb = nn.Embedding(vec.shape[0]+3, vec.shape[1], padding_idx=0)
        emb.weight.data[3:].copy_(vec)
        for p in emb.parameters():
            p.requires_grad = False
        self.embed_layer = emb
        # print('==> Inst embed layer', emb)
    
    def forward(self, sent_list):
        # sent_list [BS, max_len]
        return self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]

class IngrEmbedLayer(nn.Module):
    def __init__(self, data_dir, emb_dim):
        super(IngrEmbedLayer, self).__init__()
        path = os.path.join(data_dir, 'vocab_ingr.txt')
        with open(path, 'r') as f:
            num_ingr = len(f.read().split('\n'))
        # first three index has special meaning, see utils.py
        emb = nn.Embedding(num_ingr+3, emb_dim, padding_idx=0)
        self.embed_layer = emb
        # print('==> Ingr embed layer', emb)
    
    def forward(self, sent_list):
        # sent_list [BS, max_len]
        return self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]

class SentEncoder(nn.Module):
    def __init__(
        self, 
        data_dir,
        emb_dim,
        hid_dim, 
        with_attention=True, 
        source='inst'):
        assert source in ('inst', 'ingr')
        super(SentEncoder, self).__init__()
        if source=='inst':
            self.embed_layer = InstEmbedLayer(data_dir=data_dir, emb_dim=emb_dim) 
        elif source=='ingr':
            self.embed_layer = IngrEmbedLayer(data_dir=data_dir, emb_dim=emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)
        if with_attention:
            self.atten_layer = AttentionLayer(2*hid_dim)
        self.with_attention = with_attention
    
    def forward(self, sent_list):
        # sent_list [BS, max_len]
        x = self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]
        # print(sent_list)
        # lens = (sent_list==1).nonzero()[:,1] + 1
        lens = sent_list.count_nonzero(dim=1) + 1
        # print(lens.shape)
        sorted_len, sorted_idx = lens.sort(0, descending=True) # sorted_idx=[BS], for sorting
        _, original_idx = sorted_idx.sort(0, descending=False) # original_idx=[BS], for unsorting
        # print(sorted_idx.shape, x.shape)
        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x) # sorted_idx=[BS, max_len, emb_dim]
        sorted_inputs = x.gather(0, index_sorted_idx.long()) # sort by num_words
        packed_seq = rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)
        
        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            y, _ = rnn.pad_packed_sequence(
                out, batch_first=True) # y=[BS, max_len, 2*hid_dim], currently in WRONG order!
            unsorted_idx = original_idx.view(-1,1,1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous() # [BS, max_len, 2*hid_dim], now in correct order
            feat = self.atten_layer(output)
        else:
            _, (h,_) = self.rnn(packed_seq) # [2, BS, hid_dim], currently in WRONG order!
            h = h.transpose(0,1) # [BS, 2, hid_dim], still in WRONG order!
            # unsort the output
            unsorted_idx = original_idx.view(-1,1,1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous() # [BS, 2, hid_dim], now in correct order
            feat = output.view(output.size(0), output.size(1)*output.size(2)) # [BS, 2*hid_dim]
            
        # print('sent', feat.shape) # [BS, 2*hid_dim]
        return feat


class SentEncoderFC(nn.Module):
    def __init__(
        self, 
        data_dir,
        emb_dim,
        hid_dim, 
        with_attention=True, 
        source='inst'):
        assert source in ('inst', 'ingr')
        super(SentEncoderFC, self).__init__()
        if source=='inst':
            self.embed_layer = InstEmbedLayer(data_dir=data_dir, emb_dim=emb_dim) 
        elif source=='ingr':
            self.embed_layer = IngrEmbedLayer(data_dir=data_dir, emb_dim=emb_dim)
        self.fc = nn.Linear(emb_dim, 2*hid_dim)
        if with_attention:
            self.atten_layer = AttentionLayer(2*hid_dim)
        self.with_attention = with_attention
    
    def forward(self, sent_list):
        # sent_list [BS, max_len]
        x = self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]
        x = self.fc(x) # [BS, max_len, 2*hid_dim]
        if not self.with_attention:
            feat = x.sum(dim=1) # [BS, 2*hid_dim]
        else:
            feat = self.atten_layer(x) # [BS, 2*hid_dim]
        # print('ingredients', feat.shape)
        return feat


class DocEncoder(nn.Module):
    def __init__(self, sent_encoder, hid_dim, with_attention):
        super(DocEncoder, self).__init__()
        self.sent_encoder = sent_encoder
        self.rnn = nn.LSTM(
            input_size=2*hid_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)
        self.atten_layer_sent = AttentionLayer(2*hid_dim)
        self.with_attention = with_attention
    
    def forward(self, doc_list):
        # doc_list=[BS, max_len, max_len]
        embs = []
        lens = []
        for doc in doc_list:
            len_doc = doc.nonzero()[:,0].max().item() + 1
            lens.append(len_doc)
            emb_doc = self.sent_encoder(doc[:len_doc]) # [?, 2*hid_dim]
            embs.append(emb_doc)
        
        embs = sorted(embs, key=lambda x: -x.shape[0]) # [BS, [?, 2*hid_dim]]
        packed_seq = rnn.pack_sequence(embs)
        lens = torch.tensor(lens).long().to(embs[0].device)
        _, sorted_idx = lens.sort(0, descending=True) # sorted_idx=[BS], for sorting
        _, original_idx = sorted_idx.sort(0, descending=False) # original_idx=[BS], for unsorting

        if not self.with_attention:
            _, (h,_) = self.rnn(packed_seq) # [2, BS, hid_dim], currently in WRONG order!
            h = h.transpose(0,1) # [BS, 2, hid_dim], still in WRONG order!
            # unsort the output
            unsorted_idx = original_idx.view(-1,1,1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous() # [BS, 2, hid_dim], now in correct order
            feat = output.view(output.size(0), output.size(1)*output.size(2)) # [BS, 2*hid_dim]
        else:
            out, _ = self.rnn(packed_seq)
            y, _ = rnn.pad_packed_sequence(
                out, batch_first=True) # y=[BS, max_valid_len, 2*hid_dim], currently in WRONG order!
            unsorted_idx = original_idx.view(-1,1,1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous() # [BS, 2, hid_dim], now in correct order
            feat = self.atten_layer_sent(output)

        # print('instructions', feat.shape)
        return feat


class TextEncoder(nn.Module):
    def __init__(
        self, data_dir, text_info, hid_dim, emb_dim, z_dim, with_attention, ingr_enc_type):
        super(TextEncoder, self).__init__()
        self.text_info = text_info
        if self.text_info == '111':
            self.sent_encoder = SentEncoder(
                data_dir,
                emb_dim,
                hid_dim, 
                with_attention, 
                source='inst')
            self.doc_encoder = DocEncoder(
                self.sent_encoder, 
                hid_dim, 
                with_attention
            )
            if ingr_enc_type=='rnn':
                self.ingr_encoder = SentEncoder(
                    data_dir,
                    emb_dim,
                    hid_dim, 
                    with_attention, 
                    source='ingr')
            elif ingr_enc_type == 'fc':
                self.ingr_encoder = SentEncoderFC(
                    data_dir,
                    emb_dim,
                    hid_dim, 
                    with_attention, 
                    source='ingr')
            self.bn = nn.BatchNorm1d((2+2+2)*hid_dim)
            self.fc = nn.Linear((2+2+2)*hid_dim, z_dim)
        
        elif self.text_info == '010':
            if ingr_enc_type=='rnn':
                self.ingr_encoder = SentEncoder(
                    data_dir,
                    emb_dim,
                    hid_dim, 
                    with_attention, 
                    source='ingr')
            elif ingr_enc_type == 'fc':
                self.ingr_encoder = SentEncoderFC(
                    data_dir,
                    emb_dim,
                    hid_dim, 
                    with_attention, 
                    source='ingr')
            self.bn = nn.BatchNorm1d(2*hid_dim)
            self.fc = nn.Linear(2*hid_dim, z_dim)
    
    def forward(self, recipe_list):
        title_list, ingredients_list, instructions_list = recipe_list
        if self.text_info == '111':
            feat_title = self.sent_encoder(title_list)
            feat_ingredients = self.ingr_encoder(ingredients_list)
            feat_instructions = self.doc_encoder(instructions_list)
            feat = torch.cat([feat_title, feat_ingredients, feat_instructions], dim=1)
            feat = torch.tanh(self.fc(self.bn(feat)))
        elif self.text_info == '010':
            feat_ingredients = self.ingr_encoder(ingredients_list)
            feat = torch.tanh(self.fc(self.bn(feat_ingredients)))
        # print('recipe', feat.shape)
        return feat


class Resnet(nn.Module):
    def __init__(self, ckpt_path=None):
        super(Resnet, self).__init__()
        resnet = models.resnet50(pretrained=False)
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
        return feat