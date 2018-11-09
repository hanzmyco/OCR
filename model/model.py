import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Conv_ReLU(nn.Module):
    def __init__(self, nin, nout, ks, ss, ps, has_bn, has_bias=False):
        super(Conv_ReLU, self).__init__()
        if has_bn:
            self.subnet = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),
            )
        else:
            self.subnet = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias),
            nn.ReLU(inplace=True),
            )
        self.subnet.apply(weights_init)

    def forward(self, x):
        return self.subnet(x)


class BiLSTM(nn.Module):
    def __init__(self, nin, nhidden, nout, has_bias=False):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(nin, nhidden, bias=has_bias, bidirectional=True)
        self.embedding = nn.Linear(nhidden * 2, nout)
        self.nout = nout
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        t, b, d = rnn_out.size()
        rnn_resize = rnn_out.view(t*b, d)
        emb_out = self.embedding(rnn_resize)
        return emb_out.view(t, b, self.nout)

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_




####################################################




class BERTLayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, max_position_lengs, hidden_size, dropout_prob=0.1):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.position_embeddings = nn.Embedding(max_position_lengs, hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        ## x: n*w*c
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        
        position_ids = position_ids.unsqueeze(0).expand(x.size()[:-1])
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = position_embeddings + x
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super(BERTSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BERTLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.output = BERTSelfOutput(hidden_size, dropout_prob)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout_prob=0.1):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BERTLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(intermediate_size, hidden_size, dropout_prob)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class AttStage1(nn.Module):
    def __init__(self, feat_seq_len, out_seq_len, hidden_size):
        super(AttStage1, self).__init__()
        self.feat_seq_len = feat_seq_len
        self.out_seq_len = out_seq_len
        self.emb = BERTEmbeddings(feat_seq_len, hidden_size) ## feat + emb -> n*feat_seq_len*hidden_size
        self.ws1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ws2 = nn.Linear(hidden_size, out_seq_len, bias=False)
        self.bert = BERTLayer(feat_seq_len, 1, feat_seq_len)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(2)

    def forward(self, feat):
        n, c, h, w = feat.size()
        feat = feat.view(n, c, h*w).permute(0, 2, 1).contiguous() ## n*feat_seq_len*hidden_size
        emb_feat = self.emb(feat) ## 
        emb_proj = self.ws2(self.tanh(self.ws1(emb_feat))).permute(0, 2, 1).contiguous() ## n*feat_seq_len*hidden_size -> n*feat_seq_len*out_seq_len -> n*out_seq_len*feat_seq_len
        att_mat = self.softmax(self.bert(emb_proj))  ## n*out_seq_len*feat_seq_len
        res = (emb_feat.unsqueeze(1).expand(n, self.out_seq_len, self.feat_seq_len, c) * att_mat.unsqueeze(3).expand(n, self.out_seq_len, self.feat_seq_len, c)).sum(2)
        return res

class AttStage2(nn.Module):
    def __init__(self, out_seq_len, hidden_size, num_attention_heads, intermediate_size, num_layers, dropout_prob=0.1):
        super(AttStage2, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(BERTEmbeddings(out_seq_len, hidden_size))
            layers.append(BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob))
        self.att = nn.Sequential(*layers)

    def forward(self, x):
        return self.att(x)

class AttOutput(nn.Module):
    def __init__(self, hidden_size, nout):
        super(AttOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, nout)

    def forward(self, x):
        return self.dense(x)

############################################################################################################################
class RAN(nn.Module):
    def __init__(self, args):
        super(RAN, self).__init__()
        self.nClasses = args.nClasses
        self.cnn = self.creatModel()
        self.softmax = InferenceBatchSoftmax()
        self.att = nn.Sequential(
            AttStage1(26, 26, 512), 
            AttStage2(26, 512, 4, 512, 2),
            AttOutput(512, self.nClasses)
            )

    def creatModel(self):
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential(
            Conv_ReLU(3, nm[0], ks[0], ss[0], ps[0], False),            ## c1 false  32x100
            nn.MaxPool2d(2, 2),                                         ## p1        16x50
            Conv_ReLU(nm[0], nm[1], ks[1], ss[1], ps[1], False),        ## c2 false  16x50
            nn.MaxPool2d(2, 2),                                         ## p2        8x25
            Conv_ReLU(nm[1], nm[2], ks[2], ss[2], ps[2], True),         ## c3 true   8x25
            Conv_ReLU(nm[2], nm[3], ks[3], ss[3], ps[3], False),        ## c4 flase  8x25
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                       ## p3        4*26
            Conv_ReLU(nm[3], nm[4], ks[4], ss[4], ps[4], True),         ## c4 true   8x26
            Conv_ReLU(nm[4], nm[5], ks[5], ss[5], ps[5], False),        ## c5 flase  8x26
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                       ## p4        2*27
            Conv_ReLU(nm[5], nm[6], ks[6], ss[6], ps[6], True),         ## c6 true   1*26
        )
        return cnn

    def forward(self, x):
        ## norm data
        x = (x/255.0 - 0.5)/0.5
        cnn_out = self.cnn(x) ## N*512*1*26
        res = self.softmax(self.att(cnn_out)) ## N*26*37
        return res
