import os
import torch

def _lb(A, max_abs):
    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)
    if value0 > value1:
        x = A[:, i].conj() @ A
        return max_abs * torch.linalg.vector_norm(
            (x / torch.linalg.vector_norm(x)) @ A.H
        )
    else:
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(
            A.H @ (x / torch.linalg.vector_norm(x))
        )
def norm_lower_bound(A):
    """Cheap lower bound for the spectral norm of A."""
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)

def single_sided_whitening(G, Q, lr_param=0.5):
    m, n = G.shape
    assert(m>=n)
    V = torch.randn_like(G)/m**0.5
    A = G @ Q.T
    Bh = torch.linalg.solve_triangular(Q, V, upper=True, left=False)
    AhA = A.T @ A
    BBh = Bh.T @ Bh
    Q = Q - lr_param/norm_lower_bound(AhA + BBh) * torch.triu(AhA - BBh) @ Q
    return Q


import torch
class Nuon(torch.optim.Optimizer):
    def __init__(self, muon_params, lr=0.002, lr_param=0.1, momentum=0.95, nesterov=True,
                 whitening_prob=1.0, 
                 adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0):

        defaults = dict(lr=lr, lr_param=lr_param, momentum=momentum, nesterov=nesterov,
                        whitening_prob=whitening_prob,  # Include in defaults
                        adamw_lr_ratio=adamw_lr / lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_muon'] = True
            else:
                self.state[p]['use_muon'] = False
        for p in adamw_params:
            self.state[p]['use_muon'] = False

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group['params'] if self.state[p]['use_muon']]
            lr = group['lr']
            momentum = group['momentum']
            whitening_prob = group['whitening_prob']

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf


                m, n = g.shape
                if m<n:
                  g = g.T
                # Initialize or retrieve the Q matrix
                if 'Q' not in state:
                  state['Q'] = torch.eye(g.shape[1], device=g.device)

                Q = state['Q']

                # Decide whether to call the whitening function
                if torch.rand(1).item() < whitening_prob:                  
                    Q = single_sided_whitening(g, Q, lr_param=group['lr_param'])
                    state['Q'] = Q  # Update Q only if whitening is called

                # Use Q to whiten the gradient
                g = g @ Q.T @ Q                
                if m<n:
                  g = g.T
                g *= max(1, g.size(0) / g.size(1))**0.5

                # Apply the gradient update
                p.data.add_(g, alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if not self.state[p]['use_muon']]
            lr = group['adamw_lr_ratio'] * group['lr']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1

                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']
                scale = bias_correction1 / bias_correction2**0.5

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
