import torch
import numpy as np
from typing import Tuple, Optional
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from sklearn.manifold import TSNE

torch.set_default_dtype(torch.double)

class VAE(torch.nn.Module):
    # dim_latent is dim(Z)
    # encoder gives mean and log of std
    def _get_encoder(self, dim_in: int, dim_latent: int) -> torch.nn.Sequential:
        nn = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            # torch.nn.Dropout1d(0.2),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            # torch.nn.Dropout1d(0.2),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            # torch.nn.Linear(16, 16),
            # torch.nn.Tanh(),
            torch.nn.Linear(32, 2 * dim_latent)
        ).to(self.device)
        self.register_module('encoder_nn', nn)
        def helper_func(X: torch.Tensor, C: torch.Tensor):
            if C.ndim == 1:
                C = C[:, None]
            inp_tens = X#torch.concat((X, C), dim=1)
            return nn(inp_tens)
        return helper_func

    def _get_decoder(self, dim_latent: int, dim_out: int) -> torch.nn.Sequential:
        nn = torch.nn.Sequential(
            torch.nn.Linear(dim_latent, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            # torch.nn.Linear(16, 16),
            # torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            # torch.nn.Dropout1d(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            # torch.nn.Dropout1d(0.2),
            torch.nn.Linear(32, dim_out)
        ).to(self.device)
        self.register_module('decoder_nn', nn)
        def helper_func(X: torch.Tensor, C: torch.Tensor):
            if C.ndim == 1:
                C = C[:, None]
            inp_tens = X#torch.concat((X, C), dim=1)
            return nn(inp_tens)
        return helper_func

    def __init__(self, latent_dim: int, regular_coef: float, epochs: int, batch_num: int, l_r: float, device: torch.device,
                 encoder_dim: Optional[int]=None, decoder_dim: Optional[int]=None) -> None:
        super().__init__()
        self.device = device
        self.reg_coef = regular_coef
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        # self.sigma_nn = None
        self.l_r = l_r
        self.enc_dim = encoder_dim
        self.dec_dim = decoder_dim
        self.batch_num = batch_num
        self.epochs_num = epochs

    def _lazy_init(self, x: np.ndarray) -> None:
        enc_dim = x.shape[-1] if self.enc_dim is None else self.enc_dim
        dec_dim = self.latent_dim if self.dec_dim is None else self.dec_dim
        
        self.encoder = self._get_encoder(enc_dim, self.latent_dim)
        self.decoder = self._get_decoder(dec_dim, x.shape[-1])
        # self.sigma_nn = self._get_sigma(self.latent_dim, self.latent_dim)
        self.optimizer = torch.optim.AdamW(self.parameters(), self.l_r, weight_decay=1e-5)
        
    def get_mu_sigma(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        mu = code[:, :self.latent_dim]
        sigma = torch.exp(code[:, self.latent_dim:])
        # mu = self.encoder(x)
        # sigma = torch.exp(self.sigma_nn(mu))
        return mu, sigma
        
    def kernel(self, x_diff: torch.Tensor) -> torch.Tensor:
        C = 2 * self.latent_dim
        return C / (C + torch.sum(torch.pow(x_diff, 2), dim=-1)) # (batch)        

    def loss_fn(self, true: torch.Tensor, recon: torch.Tensor,
                mu: torch.Tensor, sigma_log: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        mse = torch.mean(torch.mean((true - recon) ** 2, dim=-1))
        kl = self.reg_coef * torch.mean(torch.sum(torch.exp(sigma_log) + mu ** 2 - sigma_log - 1, dim=-1))
        loss = mse + self.reg_coef * kl
        return loss, mse.item(), kl.item()
    
    def mmd_loss(self, x: torch.Tensor, z_latent: torch.Tensor, 
                 recon: torch.Tensor):
        assert x.shape[0] == recon.shape[0]
        
        z_smp = torch.normal(0, 1, z_latent.shape,
                             device=self.device, dtype=torch.get_default_dtype())
        
        cost = torch.mean(torch.sum(torch.pow(x - recon, 2), dim=-1))
        
        kernel = self.kernel
        z_smp_diff_mtx = z_smp[None, ...] - z_smp[:, None, :]
        K_z_smp = kernel(z_smp_diff_mtx.flatten(end_dim=-2))
        z_diff_mtx = z_latent[None, ...] - z_latent[:, None, :]
        K_z = kernel(z_diff_mtx.flatten(end_dim=-2))
        z_dis_diff = z_smp[None, ...] - z_latent[:, None, :]
        K_dis = kernel(z_dis_diff.flatten(end_dim=-2))
        
        n = z_latent.shape[0]
        diag_mask = torch.eye(n, dtype=torch.bool, device=self.device).ravel()
        K_z_smp = K_z_smp.masked_fill(diag_mask, 0)
        K_z = K_z.masked_fill(diag_mask, 0)
        
        quad_coef = self.reg_coef / (n * (n - 1))
        full_quad_coef = 2 * self.reg_coef / (n * n)
        
        regularizer = quad_coef * torch.sum(K_z_smp) + quad_coef * torch.sum(K_z) \
            - full_quad_coef * torch.sum(K_dis)
        loss = cost + regularizer
        return loss, cost.item(), regularizer.item()
        
    def gen_val_loss(self, x: torch.Tensor, c: torch.Tensor, val_num: int) -> float:
        with torch.no_grad():
            epsilon = torch.randn((val_num, self.latent_dim))
            code = torch.concat((epsilon, c), dim=-1)
            x_val = self.decoder(code)[:, None, :]
            mse_diff = torch.mean((x_val - x[None, ...]) ** 2, dim=-1)
            max = torch.amax(mse_diff, dim=-1)
        return torch.mean(max).item()
    
    # sample points from mean and std
    # if num is 1, mid dim is collapsed
    # def gen_samples(self, mu: torch.Tensor, sigma: torch.Tensor, num: int = 1) -> torch.Tensor:
    #     epsilon = torch.normal(0, 1, (mu.shape[0], num, mu.shape[1]),
    #                            device=self.device, dtype=torch.get_default_dtype())
    #     if num == 1:
    #         epsilon = epsilon[:, 0, :]
    #     else:
    #         mu = mu[:, None, :]
    #         sigma = sigma[:, None, :]
    #     return mu + epsilon * sigma

    def forward(self, x: torch.Tensor, c: torch.Tensor, samples_num: int = 1) -> torch.Tensor:
        code = self.encoder(x, c)
        mu = code[:, :self.latent_dim]
        sigma_log = code[:, self.latent_dim:]
        # sigma_log = self.sigma_nn(mu)
        eps = torch.normal(0, 1, mu.shape, device=self.device)
        z = mu + torch.exp(sigma_log) * eps
        return z

    def fit(self, x: np.ndarray, c: np.ndarray) -> 'VAE':
        assert x.ndim == 2
        self._lazy_init(x)
        X = torch.from_numpy(x).to(self.device)
        C = torch.from_numpy(c).to(self.device)

        dataset = TensorDataset(X, C)
        self.train()
        val_kw = dict()
        for e in range(self.epochs_num):
            data_loader = DataLoader(dataset, self.batch_num, True)
            prog_bar = tqdm(
                data_loader, f'Epoch {e}', unit='batch', ascii=True)
            cum_loss, cum_mse, cum_mmd = 0, 0, 0
            for i, (x_b, c_b) in enumerate(prog_bar):
                self.optimizer.zero_grad()
                ###
                code = self.encoder(x_b, c_b)
                mu = code[:, :self.latent_dim]
                sigma_log = code[:, self.latent_dim:]
                # sigma_log = self.sigma_nn(mu)
                eps = torch.normal(0, 1, mu.shape, device=self.device)
                z = mu + torch.exp(sigma_log) * eps
                x_r = self.decoder(z, c_b)
                loss, mse, regul = self.loss_fn(x_b, x_r, mu, sigma_log)
                ###
                # z = self(x_b, c_b)
                # x_r = self.decoder(z, c_b)
                # loss, mse, regul = self.mmd_loss(x_b, z, x_r)
                loss.backward()
                self.optimizer.step()
                cum_loss += loss.item()
                cum_mse += mse
                cum_mmd += regul

                prog_bar.set_postfix(Loss=cum_loss / (i + 1),
                                     MSE=cum_mse / (i + 1),
                                     KL=cum_mmd / (i + 1),
                                     **val_kw)
        self.eval()
        return self

    def predict(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(x).to(self.device)
            C = torch.from_numpy(c).to(self.device)
            z = self(X, C)
            x_r = self.decoder(z, C)
            return x_r.cpu().numpy()

    def sample(self, num: int, cond: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.normal(0, 1, (num, self.latent_dim), device=self.device)
            samples = self.decoder(z, torch.tensor(cond).to(self.device))
            return samples.cpu().numpy()
