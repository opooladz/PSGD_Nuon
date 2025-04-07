import torch
import os
import math
from typing import Optional, Tuple

def safe_solve_triangular(A: torch.Tensor, B: torch.Tensor, upper: bool = True) -> torch.Tensor:
    """Numerically stable triangular solve with dtype handling."""
    return torch.linalg.solve_triangular(
        A.float(), 
        B.float(),
        upper=upper
    ).to(A.dtype)

def norm_lower_bound(A: torch.Tensor) -> torch.Tensor:
    """Efficient spectral norm estimation with numerical safety."""
    max_abs = A.abs().max()
    if max_abs == 0:
        return torch.tensor(0.0, device=A.device)
    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)
    if value0 > value1:
        x = A[:, i].conj() @ A
        return max_abs * torch.linalg.norm((x / torch.linalg.norm(x)) @ A.H)
    else:
        x = A @ A[j].conj()
        return max_abs * torch.linalg.norm(A.H @ (x / torch.linalg.norm(x)))

def muon_style_psgd(G: torch.Tensor, 
                   GhG: Optional[torch.Tensor] = None,
                   Q: Optional[torch.Tensor] = None,
                   lr_preconditioner: float = 0.1,
                   preconditioner_update_probability: float = 0.1,
                   ghg_momentum: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized PSGD preconditioner with automatic initialization."""
    m, n = G.shape
    if m < n:
        G = G.H
        
    # Initialize on first run
    if GhG is None or Q is None:
        min_dim = min(m, n)
        GhG = G.H @ G
        Q = torch.eye(min_dim, device=G.device, dtype=G.dtype)
    
    # Whitening operation
    precondG = G @ Q.H @ Q
    
    # Update gradient second moment
    GhG = ghg_momentum * GhG + (1 - ghg_momentum) * (G.H @ G)
    
    # Probabilistic preconditioner update
    if torch.rand(1, device=G.device).item() < preconditioner_update_probability:
        invQ = safe_solve_triangular(Q, torch.eye(Q.size(0), device=G.device, dtype=Q.dtype), upper=True)
        invQhinvQ = invQ.H @ invQ
        AhA = Q @ GhG @ Q.H
        lr = lr_preconditioner / (norm_lower_bound(AhA + invQhinvQ) + 1.2e-38)
        Q = Q - lr * torch.triu(AhA - invQhinvQ) @ Q
    
    return (precondG.H, GhG, Q) if m < n else (precondG, GhG, Q)

class NuonE(torch.optim.Optimizer):
    def __init__(self, 
                 muon_params: torch.Tensor, 
                 adamw_params: Optional[torch.Tensor] = None,
                 lr: float = 3e-4,
                 lr_param: float = 0.1,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 whitening_prob: float = 0.1,
                 ghg_momentum: float = 0.9,
                 adamw_lr: float = 3e-4,
                 adamw_betas: Tuple[float, float] = (0.9, 0.999),
                 adamw_eps: float = 1e-8,
                 adamw_wd: float = 0.01):
        
        defaults = dict(
            lr=lr,
            lr_param=lr_param,
            momentum=momentum,
            nesterov=nesterov,
            whitening_prob=whitening_prob,
            ghg_momentum=ghg_momentum
        )
        adamw_defaults = dict(
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_wd
        )
        
        # Separate parameter groups
        params = [
            {'params': muon_params, **defaults},
            {'params': adamw_params or [], **adamw_defaults}
        ]
        
        super().__init__(params, defaults)
        
        # Tag parameters
        for p in muon_params:
            self.state[p]['use_muon'] = True
        if adamw_params:
            for p in adamw_params:
                self.state[p]['use_muon'] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'betas' in group:  # AdamW group
                self._adamw_step(group)
            else:  # Muon group
                self._muon_step(group)

        return loss

    def _muon_step(self, group):
        """Original Muon-style PSGD update"""
        for p in group['params']:
            if p.grad is None:
                continue
                
            state = self.state[p]
            grad = p.grad
            
            # Initialize state
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)
            
            # Momentum update
            buf = state['momentum_buffer']
            buf.mul_(group['momentum']).add_(grad)
            if group['nesterov']:
                d_p = grad.add(buf, alpha=group['momentum'])
            else:
                d_p = buf.clone()

            # Reshape for 2D processing
            original_shape = d_p.shape
            if d_p.ndim > 2:
                d_p = d_p.view(original_shape[0], -1)

            # Apply preconditioning
            precond_g, new_GhG, new_Q = muon_style_psgd(
                d_p,
                state.get('GhG'),
                state.get('Q'),
                lr_preconditioner=group['lr_param'],
                preconditioner_update_probability=group['whitening_prob'],
                ghg_momentum=group['ghg_momentum']
            )
            
            # Update state
            state['GhG'] = new_GhG
            state['Q'] = new_Q
            
            # Reshape back if needed
            if len(original_shape) > 2:
                precond_g = precond_g.view(original_shape)
            
            # Parameter update
            p.add_(precond_g, alpha=-group['lr'])

    def _adamw_step(self, group):
        """Original AdamW fallback"""
        for p in group['params']:
            if p.grad is None:
                continue
                
            grad = p.grad
            state = self.state[p]
            
            # Initialize state
            if len(state) <= 1:  # Only 'use_muon' flag exists
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            
            # Update moments
            state['step'] += 1
            beta1, beta2 = group['betas']
            
            state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
            state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # Bias correction
            bias_corr1 = 1 - beta1 ** state['step']
            bias_corr2 = 1 - beta2 ** state['step']
            
            # Weight decay
            if group['weight_decay'] != 0:
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
            
            # Update parameters
            denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_corr2)).add_(group['eps'])
            step_size = group['lr'] / bias_corr1
            p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)
