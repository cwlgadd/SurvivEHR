import numpy as np
import torch
import torch.nn as nn
import logging
import wandb

class FCNet(nn.Module):
    """Fully Connected Neural Network with Dynamic Hidden Layers"""

    def __init__(self, input_dim, hidden_dim, output_dim, output_act, device="cpu"):

        super().__init__()

        self.device = device
        logging.debug(f"FCNet: Using {self.device} as the device")

        match hidden_dim:
            case None:
                hidden_dim = []
            case int():
                hidden_dim = [hidden_dim]
            case list():
                pass
            case _:
                logging.warning(f"Invalid type {type(hidden_dim)} for hidden_dim")
                raise NotImplementedError
        
        layers = []
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

        # Compute quadrature points (scaling between 0 and t)
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
    def __init__(self, 
                 cov_dim, 
                 hidden_dim,
                 device="cpu",
                 n=15, 
                 modified=True):

        super().__init__()

        input_dim = cov_dim + 1
        self.net = CondODENet(input_dim, hidden_dim, 1, device=device, n=n, modified=modified)
        self.net = self.net.to(self.net.device)
        self.modified = modified

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, t):
        x = x.to(self.net.device)
        t = t.to(self.net.device)
        t = t.unsqueeze(1)
        return self.net.forward(x, t).squeeze()

    def predict(self, x, t):
        # wrap back for original optimize code
        return self.forward(x, t)

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
            cdf_uncens = self.net.forward(x[uncens_ids, :], t[uncens_ids, :]).squeeze()
            if not self.modified:
                cdf_uncens = cdf_uncens ** 2
            dudt_uncens = self.net.ode_mapping(x[uncens_ids, :], t[uncens_ids, :]).squeeze()
            uncensterm = (torch.log(1 - cdf_uncens + eps) + torch.log(dudt_uncens + eps)).sum()

        return -(censterm + uncensterm)

    def optimize(self, data_loader, n_epochs, logging_freq=10, data_loader_val=None,
                 max_wait=20):
        batch_size = data_loader.batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (x, t, k) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t,:]
                t_ = t[argsort_t]
                k_ = k[argsort_t]

                self.optimizer.zero_grad()
                loss = self.loss(x_,t_,k_)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                if data_loader_val is not None:
                    val_loss = 0
                    for batch_idx, (x, t, k) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t,:]
                        t_ = t[argsort_t]
                        k_ = k[argsort_t]

                        loss = self.loss(x_,t_,k_)
                        val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        print(f"best_epoch: {epoch}")
                        torch.save(self.state_dict(), "low")
                    else:
                        wait += 1

                    if wait > max_wait:
                        state_dict = torch.load("low")
                        self.load_state_dict(state_dict)
                        return

                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_loss:11.2f}")
        if data_loader_val is not None:
            state_dict = torch.load("low")
            self.load_state_dict(state_dict)
        

class ODESurvMultiple(nn.Module):
    def __init__(self, 
                 cov_dim, 
                 hidden_dim,
                 num_risks,
                 device="cpu",
                 n=15):
        super().__init__()

        input_dim = cov_dim + 1

        self.pinet = FCNet(cov_dim, hidden_dim, num_risks, nn.Softmax(dim=1), device)
        self.pinet = self.pinet.to(self.pinet.device)

        self.odenet = CondODENet(input_dim, hidden_dim, num_risks, device, n)
        self.odenet = self.odenet.to(self.odenet.device)

        self.K = num_risks

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_pi(self, x):

        return self.pinet(x)

    def forward(self, x, t):

        t = t.unsqueeze(1)
        pi = self.get_pi(x)
        preds = pi * self.odenet.forward(x, t)

        # print(f"x shape {x.shape}")
        # print(f"with preds {preds[0,:]}")

        return preds, pi

    def predict(self, x, t):
        # wrap back for original optimize code
        return self.forward(x, t)

    def loss(self, x, t, k):

        t = t.unsqueeze(1)
        eps = 1e-8

        censterm = torch.tensor(0)
        cens_ids = torch.where(k == 0)[0]
        if torch.numel(cens_ids) != 0:
            cif_cens = self.forward(x[cens_ids, :], t[cens_ids, 0])[0]
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

        # Log some parts of the loss to view in wandb 
        #    commit = False so we do not commit it until another call later commits everything 
        #    This will stop us committing all values
        # wandb.log({"DeSurv censterm": censterm}, commit=False)
        # wandb.log({"DeSurv uncensterm": uncensterm}, commit=False)
        # wandb.log({"DeSurv uncensterm sample weighted likel ": likel/x.shape[0]}, commit=False)
        # wandb.log({"DeSurv x shape": x.shape[0]}, commit=False)
        
        return -(censterm + uncensterm)    

    def optimize(self, data_loader, n_epochs, logging_freq=10, data_loader_val=None,
                 max_wait=20):
   
        batch_size = data_loader.batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (x, t, k) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t,:].to(self.odenet.device)
                t_ = t[argsort_t].to(self.odenet.device)
                k_ = k[argsort_t].to(self.odenet.device)

                self.optimizer.zero_grad()
                loss = self.loss(x_,t_,k_)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                if data_loader_val is not None:
                    val_loss = 0
                    for batch_idx, (x, t, k) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t,:].to(self.odenet.device)
                        t_ = t[argsort_t].to(self.odenet.device)
                        k_ = k[argsort_t].to(self.odenet.device)

                        loss = self.loss(x_,t_,k_)
                        val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        print(f"best_epoch: {epoch}")
                        torch.save(self.state_dict(), "low_")
                    else:
                        wait += 1

                    if wait > max_wait:
                        state_dict = torch.load("low_")
                        self.load_state_dict(state_dict)
                        return

                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_loss:11.2f}")
        if data_loader_val is not None:
            print("loading low_")
            state_dict = torch.load("low_")
            self.load_state_dict(state_dict)
            return








class CondODENetInitCond(nn.Module):
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

    def forward(self, x, t, y0, derivative=False):

        # Compute quadrature points (scaling between 0 and t)
        tau = torch.matmul(t / 2, 1 + self.u)
        tau_ = torch.flatten(tau).unsqueeze(1)
        reppedx = x.repeat_interleave(torch.tensor([self.n] * t.shape[0], dtype=torch.long, device=self.device), dim=0)

        dudt = self.ode_mapping(reppedx, tau_)
        f = dudt.reshape((*tau.shape, self.output_dim))
        integral_pred = t / 2 * ((self.w.unsqueeze(2) * f).sum(dim=1))
        integral_pred = integral_pred.to(self.device)

        # logging.info("integral_pred")
        # logging.info(integral_pred.shape)
        # logging.info(integral_pred)
        # logging.info("y0")
        # logging.info(y0.shape)
        # logging.info(y0)
        
        pred = y0 + integral_pred  # Add the initial condition
        
        if self.modified:
            return 1 - torch.exp(-pred)
        else:
            return torch.tanh(pred)

class ODESurvMultipleWithZeroTime(nn.Module):
    def __init__(self, 
                 cov_dim, 
                 hidden_dim,
                 num_risks,
                 device="cpu",
                 n=15):
        super().__init__()

        input_dim = cov_dim + 1

        self.pinet = FCNet(cov_dim, hidden_dim, num_risks, nn.Softmax(dim=1), device)
        self.pinet = self.pinet.to(self.pinet.device)

        self.odenet = CondODENetInitCond(input_dim, hidden_dim, num_risks, device, n)
        self.odenet = self.odenet.to(self.odenet.device)

        self.p0_net = FCNet(cov_dim, hidden_dim, num_risks, nn.ReLU(), device) 
        # Initialise network to have low hazard of events occuring instantly
        # for layer in self.p0_net.mapping:
        #     if isinstance(layer, nn.Linear) and layer.bias is not None:
        #         nn.init.constant_(layer.bias, -20)

        self.K = num_risks

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_pi(self, x):

        return self.pinet(x)

    def forward(self, x, t):

        t = t.unsqueeze(1)
        pi = self.get_pi(x)
        y0 = self.p0_net(x)
        preds = pi * self.odenet.forward(x, t, y0)

        # logging.info(f"x shape {x.shape}")
        # logging.info(f"with preds {preds[0,:]}")

        return preds, pi

    def predict(self, x, t):
        # wrap back for original optimize code
        return self.forward(x, t)

    def loss(self, x, t, k):

        t = t.unsqueeze(1)
        eps = 1e-8

        censterm = torch.tensor(0)
        cens_ids = torch.where(k == 0)[0]
        if torch.numel(cens_ids) != 0:
            cif_cens = self.forward(x[cens_ids, :], t[cens_ids, 0])[0]
            cdf_cens = cif_cens.sum(dim=1)
            censterm = torch.log(1 - cdf_cens + eps).sum()

        uncensterm = torch.tensor(0)
        for i in range(self.K):
            uncens_ids = torch.where(k == i + 1)[0]
            if torch.numel(uncens_ids) != 0:
                y0 = self.p0_net(x[uncens_ids, :])
                cdf_uncens = self.odenet.forward(x[uncens_ids, :], t[uncens_ids, :], y0)[:, i]
                dudt_uncens = self.odenet.ode_mapping(x[uncens_ids, :], t[uncens_ids, :])[:, i]
                pi = self.get_pi(x[uncens_ids, :])[:, i]

                likel = (torch.log(1 - cdf_uncens + eps) + torch.log(dudt_uncens + eps) + torch.log(pi + eps)).sum()
                uncensterm = uncensterm + likel

        # Log some parts of the loss to view in wandb 
        #    commit = False so we do not commit it until another call later commits everything 
        #    This will stop us committing all values
        # wandb.log({"DeSurv censterm": censterm}, commit=False)
        # wandb.log({"DeSurv uncensterm": uncensterm}, commit=False)
        # wandb.log({"DeSurv uncensterm sample weighted likel ": likel/x.shape[0]}, commit=False)
        # wandb.log({"DeSurv x shape": x.shape[0]}, commit=False)
        # wandb.log({"DeSurv p0 min": y0.min()}, commit=False)
        # wandb.log({"DeSurv p0 max": y0.max()}, commit=False)
        
        return -(censterm + uncensterm)    

    def optimize(self, data_loader, n_epochs, logging_freq=10, data_loader_val=None,
                 max_wait=20):
   
        batch_size = data_loader.batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (x, t, k) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t,:].to(self.odenet.device)
                t_ = t[argsort_t].to(self.odenet.device)
                k_ = k[argsort_t].to(self.odenet.device)

                self.optimizer.zero_grad()
                loss = self.loss(x_,t_,k_)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                if data_loader_val is not None:
                    val_loss = 0
                    for batch_idx, (x, t, k) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t,:].to(self.odenet.device)
                        t_ = t[argsort_t].to(self.odenet.device)
                        k_ = k[argsort_t].to(self.odenet.device)

                        loss = self.loss(x_,t_,k_)
                        val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        print(f"best_epoch: {epoch}")
                        torch.save(self.state_dict(), "low_")
                    else:
                        wait += 1

                    if wait > max_wait:
                        state_dict = torch.load("low_")
                        self.load_state_dict(state_dict)
                        return

                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_loss:11.2f}")
                    
        if data_loader_val is not None:
            print("loading low_")
            state_dict = torch.load("low_")
            self.load_state_dict(state_dict)
            return