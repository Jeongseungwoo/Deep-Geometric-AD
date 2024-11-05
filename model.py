# ODE-RGRU extension version
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torchdiffeq import odeint as odeint

import numpy as np
import math


class ODERGRU_imputation(nn.Module):
    def __init__(self, input_dim, latents, rgru_hid_size, n_layers, ode_units, reg_weight, label_weight):
        super(ODERGRU_imputation, self).__init__()
        self.input_dim = input_dim
        self.latents = latents
        self.rgru_hid_size = rgru_hid_size
        self.n_layers = n_layers
        self.ode_units = ode_units
        self.classes = True

        self.reg_weight = reg_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder(self.rgru_hid_size)

        self.rgru_d = RGRUCell(self.latents, self.rgru_hid_size, True)
        self.rgru_l = RGRUCell(self.latents * (self.latents - 1) // 2,
                               self.rgru_hid_size * (self.rgru_hid_size - 1) // 2, False)

        self.odefunc = ODEFunc(self.rgru_hid_size * (self.rgru_hid_size + 1) // 2, self.n_layers, self.ode_units)

        if self.classes == True:
            self.out_cls = nn.Linear(self.rgru_hid_size * (self.rgru_hid_size + 1) // 2, 3)

    def forward(self, data, criterion_reg, criterion_cls, multi_flag=True):
        s = data['data'].permute(0, 2, 1)[:, :, :10]  # batch, channel, time

        mask = data['mask'].permute(0, 2, 1)[:, :, :10]
        label = data['label'][:, :10]

        label = label.contiguous().view(-1, 1)
        label_indicator = torch.ones_like(label)
        label_indicator[torch.where(label == -2)[0]] = 0

        h_d = torch.ones(s.shape[0], self.rgru_hid_size, device=s.device)
        h_l = torch.zeros(s.shape[0], self.rgru_hid_size * (self.rgru_hid_size - 1) // 2, device=s.device)
        times = torch.from_numpy(np.arange(s.shape[2] + 1)).float().to(s.device)

        output_reg, output_cls = [], []
        output_probs = []
        out = []

        for i in range(s.shape[2]):
            observation = mask[:, :, i].min(dim=1).values != 0.0
            if observation.any():
                x_d, x_l = self.encoder(s[:, :, i], mask[:, :, i])  # [b, c]
            else:
                s_ = output_reg[-1]
                s_h = torch.where(mask[:, :, i].bool(), s[:, :, i], s_)
                x_d, x_l = self.encoder(s_h, torch.ones(mask.shape[0], mask.shape[1], device=s.device))
            hp = odeint(self.odefunc,
                        torch.cat((h_d.log(), h_l), dim=1),
                        times[i:i + 2],
                        rtol=1e-4,
                        atol=1e-5,
                        method='euler')[1]
            h_d = hp[:, :self.rgru_hid_size].tanh().exp()
            h_l = hp[:, self.rgru_hid_size:]

            h_d = self.rgru_d(x_d, h_d)
            h_l = self.rgru_l(x_l, h_l)

            h = torch.cat((h_d.log(), h_l), dim=1)

            if multi_flag == True:
                y_cls = self.out_cls(h)
                output_prob = torch.softmax(y_cls, dim=1)
                output_cls.append(y_cls.unsqueeze(dim=1))
                output_probs.append(output_prob.unsqueeze(dim=1))

            s_ = self.decoder(h)  # regression parts
            output_reg.append(torch.cat(s_, dim=1))

            out.append(h)

        output_reg = torch.stack(output_reg, dim=2)  # batch, channel, time
        if multi_flag == True:
            output_cls = torch.cat(output_cls, dim=1)
            output_probs = torch.cat(output_probs, dim=1)

        y_reg_loss = criterion_reg(output_reg[:, :6, :] * data['mask'].permute(0, 2, 1)[:, :6, 1:], data['data'].permute(0, 2, 1)[:, :6, 1:] * data['mask'].permute(0, 2, 1)[:, :6, 1:]) # MRI-biomarkers
        y_mmse_loss = criterion_reg(output_reg[:, 6:7, :] * data['mask'].permute(0, 2, 1)[:, 6:7, 1:], data['data'].permute(0, 2, 1)[:, 6:7, 1:] * data['mask'].permute(0, 2, 1)[:, 6:7, 1:]) # MMSE
        y_ad11_loss = criterion_reg(output_reg[:, 7:8, :] * data['mask'].permute(0, 2, 1)[:, 7:8, 1:], data['data'].permute(0, 2, 1)[:, 7:8, 1:] * data['mask'].permute(0, 2, 1)[:, 7:8, 1:]) # ADAS-cog11
        y_ad13_loss = criterion_reg(output_reg[:, 8:9, :] * data['mask'].permute(0, 2, 1)[:, 8:9, 1:], data['data'].permute(0, 2, 1)[:, 8:9, 1:] * data['mask'].permute(0, 2, 1)[:, 8:9, 1:]) # ADAS-cog13
        if multi_flag == True:
            y_cls_loss = criterion_cls(output_probs.contiguous().view(-1, 3), label.squeeze().long())

        return {
            "loss": self.reg_weight*(y_mmse_loss) + 0.5*(y_ad11_loss + y_ad13_loss + y_reg_loss) + self.label_weight*y_cls_loss,
            "predict": output_probs.contiguous().view(-1, 3),
            "predict_feature": output_reg[:, :6, :].permute(0, 2, 1).contiguous().view(-1, 6) ,
            'labels': label,
            "is_train": label_indicator,
            'shift_data': data['data'][:, 1:, :6], # b, t, c
            'shift_mask': data['mask'][:, 1:, :6],
            'predict_mmse': output_reg[:, 6:7, :].permute(0, 2, 1).contiguous(),
            'predict_ad11': output_reg[:, 7:8, :].permute(0, 2, 1).contiguous(),
            'predict_ad13': output_reg[:, 8:9, :].permute(0, 2, 1).contiguous()
        }

    def run_on_batch(self, data, optimizer, criterion_reg, criterion_cls, multi_flag=True, epoch=None):
        ret = self(data, criterion_reg, criterion_cls, multi_flag)
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

class Encoder(nn.Module):
    def __init__(self, latents):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, latents, kernel_size=1, stride=1), # kernel_size=1
            nn.BatchNorm1d(latents),
            nn.LeakyReLU(),
        )
    # tanh, elu, selu
    def forward(self, x, mask):
        b, n = x.shape
        x = x * mask
        x.unsqueeze_(1)
        observation = mask.min(dim=1).values != 0.0
        for layer in self.layers:
            x = layer(x)
        x = torch.where(observation.unsqueeze(1).unsqueeze(1), x, torch.tensor(0., device=x.device))
        x_d = []
        x_l = []
        for i in range(b):
            if x[i].max() != 0.0:
                cov = oas_cov(x[i].transpose(-1, -2))
                d, l = self.chol_de(cov.unsqueeze(0))
                x_d.append(d.squeeze())
                x_l.append(l.squeeze())
            else:
                x_d.append(torch.ones(x.shape[1], device=x.device))
                x_l.append(torch.zeros(x.shape[1] * (x.shape[1] - 1) // 2, device=x.device))
        return torch.stack(x_d, dim=0).squeeze(), torch.stack(x_l, dim=0).squeeze()

    def chol_de(self, x):
        b, n, n = x.shape
        x = x.reshape(-1, n, n)
        L = x.cholesky()
        d = x.new_zeros(b, n)
        l = x.new_zeros(b, n * (n - 1) // 2)
        for i in range(b):
            d[i] = L[i].diag()
            l[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n)], dim=1)[0]
        return d.reshape(b, -1), l.reshape(b, -1)


class Decoder(nn.Module):
    def __init__(self, rgru_hid_size):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(rgru_hid_size * (rgru_hid_size + 1) // 2, 6),  # MRI-biomarkers
            nn.Linear(rgru_hid_size * (rgru_hid_size + 1) // 2, 1),  # MMSE
            nn.Linear(rgru_hid_size * (rgru_hid_size + 1) // 2, 1),  # ADAS-cog11
            nn.Linear(rgru_hid_size * (rgru_hid_size + 1) // 2, 1)  # ADAS-cog13
        )

    def forward(self, x):
        output = []
        for layer in self.layers:
            output.append(layer(x))
        return output

########

class ODEFunc(nn.Module):
    def __init__(self, n_inputs, n_layers, n_units):
        super(ODEFunc, self).__init__()
        self.gradient_net = odefunc(n_inputs, n_layers, n_units)

    def forward(self, t_local, y, backwards=False):
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        return self.get_ode_gradient_nn(t_local, y)

class odefunc(nn.Module):
    def __init__(self, n_inputs, n_layers, n_units):
        super(odefunc, self).__init__()
        self.Layers = nn.ModuleList()
        self.Layers.append(nn.Linear(n_inputs, n_units))
        for i in range(n_layers):
            self.Layers.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Linear(n_units, n_units)
                )
            )
        self.Layers.append(nn.LeakyReLU())
        self.Layers.append(nn.Linear(n_units, n_inputs))

    def forward(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x

class RGRUCell(nn.Module):
    """
    An implementation of RGRUCell.
    """
    def __init__(self, input_size, hidden_size, diag=True):
        super(RGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.diag = diag
        if diag:
            layer = PosLinear
            self.nonlinear = nn.Softplus()
        else:
            layer = nn.Linear
            self.nonlinear = nn.Tanh()
        self.x2h = layer(input_size, 3 * hidden_size, bias=False)
        self.h2h = layer(hidden_size, 3 * hidden_size, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_size * 3))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        b_r, b_i, b_n = self.bias.chunk(3, 0)

        if self.diag:
            resetgate = (b_r.abs() * (i_r.log() + h_r.log()).exp()).sigmoid()
            inputgate = (b_i.abs() * (i_i.log() + h_i.log()).exp()).sigmoid()
            newgate = self.nonlinear((b_n.abs() * (i_n.log() + (resetgate * h_n).log()).exp()))
            hy = (newgate.log() * (1 - inputgate) + inputgate * hidden.log()).exp()
        else:
            resetgate = (i_r + h_r + b_r).sigmoid()
            inputgate = (i_i + h_i + b_i).sigmoid()
            newgate = self.nonlinear(i_n + (resetgate * h_n) + b_n)
            hy = newgate + inputgate * (hidden - newgate)

        return hy

class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))

    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight))

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def oas_cov(X):
    n_samples, n_features = X.shape
    emp_cov = cov(X)
    mu = emp_cov.diag().sum() / n_features

    alpha = (emp_cov ** 2).mean()
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    shrinkage = 1. if den == 0 else torch.minimum((num / den), mu.new_ones(1))
    shrunk_cov = (1. - shrinkage) * emp_cov
    shrunk_cov.flatten()[::n_features + 1] += shrinkage * mu

    return shrunk_cov
