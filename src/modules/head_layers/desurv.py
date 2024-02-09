import numpy as np
import torch
import torch.nn as nn
import logging

class FCNet(nn.Module):
    """Fully Connected Neural Network with Dynamic Hidden Layers"""

    def __init__(self, input_dim, hidden_dim, output_dim, output_act, device="cpu"):

        super().__init__()

        self.device = device
        logging.debug(f"FCNet: Using {self.device} as the device")

        layers = []
        input_dim = input_dim
        for lyr in range(len(hidden_dim)):
            layers.append(nn.Linear(input_dim, hidden_dim[lyr]))
            layers.append(nn.ReLU())
            input_dim = hidden_dim[lyr]

        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(output_act)
        layers = nn.ModuleList(layers)
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)


class CondODENet(nn.Module):
    """Conditional ODE Neural Network"""

    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu", n=15, modified=True):

        super().__init__()

        self.device = device #torch.device("cuda:0" if device == "gpu" and torch.cuda.is_available() else "cpu")
        logging.debug(f"CondODENet: Using {self.device} as the device")
        self.modified = modified

        self.output_dim = output_dim
        self.n = n
        u, w = np.polynomial.legendre.leggauss(n)
        self.u = nn.Parameter(torch.tensor(u, device=self.device, dtype=torch.float32)[None, :], requires_grad=False)
        self.w = nn.Parameter(torch.tensor(w, device=self.device, dtype=torch.float32)[None, :], requires_grad=False)

        self.BaseNet = FCNet(input_dim, hidden_dim, output_dim, nn.Softplus(), device)
        self.BaseNet = self.BaseNet.to(self.BaseNet.device)

    def ode_mapping(self, x, t):

        z = torch.cat((x, t), 1)
        return self.BaseNet(z)

    def forward(self, x, t, derivative=False):
        
        tau = torch.matmul(t / 2, 1 + self.u)
        tau_ = torch.flatten(tau).unsqueeze(1)
        reppedx = x.repeat_interleave(torch.tensor([self.n] * t.shape[0], dtype=torch.long, device=self.device), dim=0)

        dudt = self.ode_mapping(reppedx, tau_)
        f = dudt.reshape((*tau.shape, self.output_dim))
        pred = t / 2 * ((self.w.unsqueeze(2) * f).sum(dim=1))
        pred = pred.to(self.device)

        if self.modified:
            return 1 - torch.exp(-pred)
        else:
            return torch.tanh(pred)
            

class ODESurvSingle(nn.Module):
    def __init__(self, cov_dim, hidden_dim, device="cpu", n=15, modified=True):

        super().__init__()

        input_dim = cov_dim + 1
        self.net = CondODENet(input_dim, hidden_dim, 1, device=device, n=n, modified=modified)
        self.net = self.net.to(self.net.device)
        self.modified = modified

    def forward(self, x, t):
        x = x.to(self.net.device)
        t = t.to(self.net.device)
        t = t.unsqueeze(1)
        return self.net.forward(x, t).squeeze()

    def loss(self, x, t, k):
        # print(f"x: {x.shape}, t:{t.shape} k:{k.shape}")
        x = x.to(self.net.device)
        t = t.to(self.net.device)
        k = k.to(self.net.device)

        t = t.unsqueeze(1)
        eps = 1e-8

        censterm = torch.tensor(0)
        cens_ids = torch.where(k == 0)[0]
        if torch.numel(cens_ids) != 0:
            cdf_cens = self.net.forward(x[cens_ids, :], t[cens_ids, :]).squeeze()
            censterm = torch.log(1 - cdf_cens + eps).sum()

        uncensterm = torch.tensor(0)
        uncens_ids = torch.where(k == 1)[0]
        if torch.numel(uncens_ids) != 0:
            cdf_uncens = self.net.forward(x[uncens_ids, :], t[uncens_ids, :], derivative=True).squeeze()
            if not self.modified:
                cdf_uncens = cdf_uncens ** 2
            dudt_uncens = self.net.ode_mapping(x[uncens_ids, :], t[uncens_ids, :]).squeeze()
            uncensterm = (torch.log(1 - cdf_uncens + eps) + torch.log(dudt_uncens + eps)).sum()

        return -(censterm + uncensterm)


class ODESurvMultiple(nn.Module):
    def __init__(self, cov_dim, hidden_dim_fc, hidden_dim_ode, num_risks, device=“cpu”, n=15):
        super().__init__()

        input_dim = cov_dim + 1

        self.pinet = FCNet(cov_dim, hidden_dim_fc, num_risks, nn.Softmax(dim=1), device)
        self.pinet = self.pinet.to(self.pinet.device)

        self.odenet = CondODENet(input_dim, hidden_dim_ode, num_risks, device, n)
        self.odenet = self.odenet.to(self.odenet.device)

        self.K = num_risks

    def get_pi(self, x):

        return self.pinet(x)

    def predict(self, x, t):

        t = t.unsqueeze(1)
        pi = self.get_pi(x)
        preds = pi * self.odenet.forward(x, t)

        return preds, pi

    def loss(self, x, t, k):

        t = t.unsqueeze(1)
        eps = 1e-8

        censterm = torch.tensor(0)
        cens_ids = torch.where(k == 0)[0]
        if torch.numel(cens_ids) != 0:
            cif_cens = self.predict(x[cens_ids, :], t[cens_ids, 0])[0]
            cdf_cens = cif_cens.sum(dim=1)
            censterm = torch.log(1 - cdf_cens + eps).sum()

        uncensterm = torch.tensor(0)
        for i in range(self.K):
            uncens_ids = torch.where(k == i + 1)[0]
            if torch.numel(uncens_ids) != 0:
                cdf_uncens = self.odenet.forward(x[uncens_ids, :], t[uncens_ids, :])[:, i]
                dudt_uncens = self.odenet.ode_mapping(x[uncens_ids, :], t[uncens_ids, :])[:, i]
                pi = self.get_pi(x[uncens_ids, :])[:, i]

                likel = (torch.log(1 - cdf_uncens + eps) + torch.log(dudt_uncens + eps) + torch.log(pi + eps)).sum()
                uncensterm = uncensterm + likel

        return -(censterm + uncensterm)    
