import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class Config(object):
    def __init__(self, dataset):
        self.data_path = dataset 
        self.class_list = ['positive', 'negative', 'neutral']                               
        self.save_path = 'saved_dict/bert.ckpt'
        self.device = torch.device('cuda:1')  
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 3                       
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '/home/liuyichen/finbert-tone'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.1
        self.rnn_hidden = 256
        self.num_layers = 2

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, return_dict=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        encoder_out = outputs.last_hidden_state
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out