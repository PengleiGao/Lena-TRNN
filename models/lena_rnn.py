import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from layers.energy_score import Block
from layers.Embed import DataEmbedding
import matplotlib.pyplot as plt
import os


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.num_steps = configs.num_step
        self.cof = configs.score_cof
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.global_trans = nn.Sequential(*[Block(configs) for _ in range(configs.d_layers)])
        if self.task_name == 'long_term_forecast':
            self.predict_linear1 = nn.Linear(self.seq_len, self.pred_len)
            self.predict_linear2 = nn.Linear(self.pred_len, self.pred_len)
            self.proj = nn.Linear(configs.d_model, configs.c_out)
            self.rnn = nn.GRUCell(configs.enc_in + configs.enc_in, configs.d_model)
            self.loss_type = nn.L1Loss()
        if self.task_name == 'imputation':
            self.rnn = nn.GRUCell(configs.enc_in + configs.enc_in, configs.d_model)
            self.proj = nn.Linear(configs.d_model, configs.c_out)
            self.loss_type = nn.L1Loss()

        self.score_head = nn.Linear(configs.d_model, 1, bias=False)
        self.alpha = nn.Embedding(1, 1)  # torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        nn.init.constant(self.alpha.weight, 0.1)


    def long_forecast(self, x_enc, x_mark_enc, batch_y, y_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        f_dim = -1 if self.configs.features == 'MS' else 0

        # energy_update
        pre_enc = self.predict_linear1(x_enc.permute(0,2,1)).permute(0,2,1)
        local_loss = 0
        for _ in range(self.num_steps):
            corrupt_enc = Variable(pre_enc.detach(), requires_grad=True)
            corrupt_enc = self.predict_linear2(corrupt_enc.permute(0, 2, 1)).permute(0, 2, 1)
            # embedding
            emb_out = self.enc_embedding(corrupt_enc, y_mark_enc[:, -self.pred_len:, :])  # [B,T,C]
            emb_out = self.global_trans(emb_out)
            emb_out = self.proj(emb_out)
            hx = torch.randn(x_enc.size(0), self.d_model).cuda()
            output = []
            for i in range(self.pred_len):
                rnn_inp = corrupt_enc[:, i, :]
                condition_inp = emb_out[:, i, :]
                hx = self.rnn(torch.cat([rnn_inp, condition_inp], dim=-1), hx)
                energy_score = self.score_head(hx)*self.cof
                inp_grad = torch.autograd.grad([energy_score.sum()], [rnn_inp], create_graph=True)[0]
                alpha = self.alpha.weight.squeeze(1)
                rnn_inp = rnn_inp - alpha * inp_grad
                output.append(rnn_inp)
            output = torch.stack(output, dim=0).permute(1, 0, 2)
            pre_enc = output

            # De-Normalization from Non-stationary Transformer
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

            local_loss += self.loss_type(output[:, -self.pred_len:, f_dim:], batch_y[:, -self.pred_len:, f_dim:].cuda())

        return output, local_loss


    def imputation(self, x_enc, x_mark_enc, target, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # energy_update
        local_loss = 0
        for _ in range(self.num_steps):
            corrupt_enc = Variable(x_enc.detach(), requires_grad=True)

            emb_out = self.enc_embedding(corrupt_enc, x_mark_enc)  # [B,T,C]
            emb_out = self.global_trans(emb_out)
            emb_out = self.proj(emb_out)
            hx = torch.randn(x_enc.size(0), self.d_model).cuda()
            output = []
            for i in range(self.seq_len):
                rnn_inp = corrupt_enc[:,i,:]
                condition_inp = emb_out[:,i,:]
                hx = self.rnn(torch.cat([rnn_inp,condition_inp],dim=-1), hx)
                energy_score = self.score_head(hx)*self.cof
                inp_grad = torch.autograd.grad([energy_score.sum()], [rnn_inp], create_graph=True)[0]
                alpha = self.alpha.weight.squeeze(1)
                rnn_inp = rnn_inp - alpha*inp_grad
                output.append(rnn_inp)
            output = torch.stack(output,dim=0).permute(1,0,2)
            x_enc = output


            # De-Normalization from Non-stationary Transformer
            output = output * \
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1, self.seq_len, 1))
            output = output + \
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, self.seq_len, 1))
            local_loss += self.loss_type(output[mask == 0], target[mask == 0])

        return output, local_loss


    def forward(self, x_enc, x_mark_enc, batch_y, batch_y_mark, mask=None):
        if self.task_name == 'long_term_forecast':
            out, total_loss = self.long_forecast(x_enc, x_mark_enc, batch_y, batch_y_mark)
            return out[:, -self.pred_len:, :], total_loss  # [B, L, D]
        if self.task_name == 'imputation':
            out, total_loss = self.imputation(x_enc, x_mark_enc, batch_y, mask)
            return out, total_loss  # [B, L, D]
        return None
