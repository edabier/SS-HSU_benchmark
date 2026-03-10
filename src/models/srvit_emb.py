import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import src.models.srvit as srvit
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"-----------------------------------------------------------------------------------------------------------"
"SRViT: Self-Supervised Relation-Aware Vision Transformer for Hyperspectral Unmixing"
"Our work uses Python 3.8 and Pytorch 2.0 + cudnn 11.8"
" Authors: Yuanchao Su, Lianru Gao, Antonio Plaza, Mengying Jiang, Xu Sun, and Guang Yang "
" Mar. 2023 "
"-----------------------------------------------------------------------------------------------------------"

def norm2squ(x):
    """Return L2norm^2"""
    return torch.sum(torch.pow(x, exponent=2))

class embedding(nn.Module):
    def __init__(self, B, c):
        super().__init__()
        self.model = srvit.AE().to(device)
        self.encoder = nn.Sequential(
            nn.Linear(B, B // 2, bias=True),
            nn.ReLU(),
            nn.Linear(B // 2, B // 4, bias=True),
            nn.ReLU(),
            nn.Linear(B // 4, c, bias=True),
            nn.Softmax(dim=1),
        )
        self.decoder = nn.Linear(c, B, bias=False)

    def forward(self, y):
        code = self.encoder(y)
        output = self.decoder(code)
        return code, output

class TimeReminder:
    def __init__(self):
        self.start = datetime.now()

    def remind(self, now_cnt, max_cnt):
        return (datetime.now()-self.start) / (now_cnt+1) * (max_cnt - now_cnt + 1e-5)

    def consuming(self):
        return datetime.now() - self.start

class reconsitution:
    def __init__(self, B, c, init_edm=None, height=None, width=None, version='com', seed=30):
        super(reconsitution, self).__init__()
        if seed:
            torch.manual_seed(seed)
        self.lr_decay = 1.
        self.reg_decay = 1e-7
        self.unfreeze_time = 0.9
        self.input_shape = (B,)
        if version == 'com':
            self.model = embedding(B, c).to(device)
            self.Br_decay = 0.9
            self.reg_norm: list = [2, 2, 2]
            self.reg_layer: list = ['encoder.0.weight', 'encoder.2.weight', 'encoder.4.weight']
            self.edm_layer = 'decoder.weight'
            self._load_endmember(edm=init_edm, layer_name=self.edm_layer)
            self.frozen_layer = [self.edm_layer]
            self.Boss_func = self._loss_function(mode='sad')
            h = torch.from_numpy(srvit.smooth_matrix(height=height, width=width)).float()
            self.s0 = torch.abs(h) > 0.1
            self.s1 = h < -0.1
            self.s2 = torch.eye(height*width)
            self.h = h.to(device)
            self.h.requires_grad = True
            self.s0 = self.s0.to(device)
            self.s1 = self.s1.to(device)
            self.s2 = self.s2.to(device)
            self.unfreeze_time = 0.9
            self.opt = torch.optim.Adam([{'params': self.model.encoder.parameters(), 'lr': 1e-3},
                                         {'params': self.model.decoder.parameters(), 'lr': 1e-2},
                                         {'params': self.h, 'lr': 1e-1}])
        else:
            raise Exception('Version Error. com/cnn')
        self.B, self.c = B, c
        self.new_edm = 0
        self.height = height
        self.width = width
        self.version = version
        self.edm = init_edm
        self.timer = TimeReminder()

    def _load_endmember(self, edm, layer_name):
        edm = torch.from_numpy(edm)
        model_dict = self.model.state_dict()
        model_dict[layer_name] = edm
        self.model.load_state_dict(model_dict)

    def _freeze(self):
        if self.frozen_layer is None:
            return
        for name, value in self.model.named_parameters():
            if name in self.frozen_layer:
                value.requires_grad = False

    def _unfreeze(self):
        if self.frozen_layer is None:
            return
        for name, para in self.model.named_parameters():
            if name in self.frozen_layer:
                para.requires_grad = True

    def _regularization_loss(self, layer_name=None, weight_decay=1e-5, p=None):
        if layer_name is None:
            return 0
        r_loss = 0
        for name, param in self.model.named_parameters():
            if name in layer_name:
                norm_num = p[layer_name.index(name)]
                if norm_num == 1 or norm_num == 2:
                    r_loss += weight_decay * torch.norm(param, p=norm_num)
        return r_loss

    def _special_loss(self, code, output):
        if self.version == 'com':
            return 1e-6 * norm2squ(torch.mm(self.h*self.s0, code)) +\
                   5 * (norm2squ(torch.sum(self.h*self.s1, dim=0)+torch.tensor(1).cpu()) +
                        norm2squ(self.h*self.s2-self.s2))
        else:
            return 0

    @staticmethod
    def _loss_function(mode='sad'):
        if mode == 'sad':
            return lambda output, target: \
                torch.mean(torch.acos(torch.sum(output * target, dim=-1) /
                           (torch.norm(output, dim=-1, p=2)*torch.norm(target, dim=-1, p=2))))
        else:
            raise Exception('Version Error! sad/')

    def fit(self, y, max_iter=200):
        pix = torch.from_numpy(y.T).float().to(device)
        self._freeze()
        loss_record = 1e5
        epoch = 0
        while epoch < max_iter:
            code, output = self.model(pix)
            base_loss = self.Boss_func(output, pix)
            r_loss = self._regularization_loss(layer_name=self.reg_layer, p=self.reg_norm, weight_decay=self.reg_decay)
            s_loss = self._special_loss(code, output)
            loss = base_loss + r_loss + s_loss
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=1)
            self.opt.step()
            epoch += 1
            if (epoch+1) % 200 == 0:
                for i, dic in enumerate(self.opt.param_groups):
                    dic['lr'] *= self.lr_decay
        m = None
        for name, para in self.model.named_parameters():
            if name == self.edm_layer:
                m = para.cpu().data.numpy()
        a = code.cpu()
        if len(a.shape) == 2:
            a = a.data.numpy().T
        else:
            a = a.data.numpy()
            a = np.squeeze(a)
            a = a.reshape(a.shape[0], -1)
        a = a / np.sum(a, axis=0)
        if m is not None:
            self.new_edm = m[:self.B]
        else:
            self.new_edm = self.edm[:self.B]
        return a

class weighconst:
    def __init__(self, y, y0, x, x0, qw, pw, ic, my, p):
        self.y = y
        self.y0 = y0
        self.x = x
        self.x0 = x0
        self.qw = qw
        self.pw = pw
        self.ic = ic
        self.my = my
        self.p = p

    def run(self):
        width, length = self.y.shape
        d0 = np.sqrt(np.sum((self.y - self.x @ self.x0) ** 2) / (width * length))
        fit = np.zeros(200)
        iteration = 0
        fit[iteration] = d0
        obj = np.zeros(200)
        obj[iteration] = fit[iteration]
        err = np.inf
        for iteration in range(1, 200):
            if err >= 1e-6:
                eet = np.kron(self.y0 @ self.y0.T, np.eye(self.p))
                b = np.kron(np.eye(self.p), np.ones((1, self.p)))
                qm = np.sum(np.linalg.inv(self.y0 @ self.y0.T) @ self.y0, axis=1)
                lam_quad = 1e-6
                h = lam_quad * np.eye(self.p ** 2)
                mu = self.p * 1000 / length
                f = h + mu * eet
                fw = np.linalg.inv(f)
                gw = fw @ b.T @ np.linalg.inv(b @ fw @ b.T)
                qm_aux = gw @ qm
                gw = fw - gw @ b @ fw
                zw = self.qw @ self.y0
                bk = np.zeros_like(zw)
                iw = np.linalg.inv(self.qw)
                g = -iw.T
                g = g.ravel()
                baux = h @ self.qw.ravel() - g
                e1 = iw * np.sqrt(self.p)
                e1 = e1[:self.p - 1, :]
                e1 = self.pw[:, :self.p - 1] @ self.ic @ e1
                e1 = self.pw.T @ e1
                for i in range(1, 100):
                    dq_aux = zw + bk
                    dtz_b = dq_aux @ self.y0.T
                    dtz_b = dtz_b.ravel()
                    b0 = baux + mu * dtz_b
                    q1 = gw @ b0 + qm_aux
                    self.qw = np.reshape(q1, (self.p, self.p))
                e1 = e1 * np.sqrt(self.p)
                e1 = e1[:self.p - 1, :]
                e1 = self.pw[:, :self.p - 1] @ self.ic @ e1
                e1 = e1 + self.my
                self.x = self.x * ((self.y @ self.x0.T) / (self.x @ self.x0 @ self.x0.T))
                Y_re = self.x @ self.x0
                d1 = np.sqrt(np.sum((self.y - Y_re) ** 2) / (width * length))
                iteration += 1
                fit[iteration - 1] = d1
                obj[iteration - 1] = fit[iteration - 1]
                err = np.abs((obj[iteration - 2] - obj[iteration - 1]) / obj[iteration - 1])
        return self.x
