import torch
import os

def safe_solve_triangular(A, B, upper=True, left=True):
    """Wrapper for solve_triangular that handles bfloat16 via float32 conversion."""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    result = torch.linalg.solve_triangular(A, B, upper=upper, left=left)
    return result.to(A.dtype)

def _lb(A, max_abs):
    """Helper for norm_lower_bound."""
    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)
    if value0 > value1:
        x = A[:, i].conj() @ A
        return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A.H)
    else:
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(A.H @ (x / torch.linalg.vector_norm(x)))

def norm_lower_bound(A):
    """Cheap lower bound for spectral norm."""
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)

def muon_style_psgd(G, GhG, Q, lr_preconditioner=0.1, preconditioner_update_probability=1, ghg_momentum=0.9):
    """
    Muon-style PSGD with configurable GhG momentum.
    
    Args:
        G: Gradient matrix (m × n)
        GhG: EMA of G.H @ G
        Q: Current preconditioner
        lr_preconditioner: Learning rate for Q update
        preconditioner_update_probability: Probability of updating Q
        ghg_momentum: Momentum for GhG EMA (default 0.9)
        
    Returns:
        Tuple of (whitened gradient, updated GhG, updated Q)
    """
    m, n = G.shape
    if m < n:
        G = G.H
        
    precondG = G @ Q.H @ Q
    GhG = ghg_momentum * GhG + (1-ghg_momentum) * G.H @ G
    
    if torch.rand([]) < preconditioner_update_probability:
        invQ = safe_solve_triangular(
            Q, 
            torch.eye(min(m,n), device=G.device, dtype=G.dtype), 
            upper=True
        )
        invQhinvQ = invQ.H @ invQ
        AhA = Q @ GhG @ Q.H
        lr = lr_preconditioner / (norm_lower_bound(AhA + invQhinvQ) + 1.2e-38)
        Q = Q - lr * torch.triu(AhA - invQhinvQ) @ Q
    
    return (precondG.H, GhG, Q) if m < n else (precondG, GhG, Q)

class NuonE(torch.optim.Optimizer):
    def __init__(self, muon_params, lr=3e-4, lr_param=0.1, momentum=0.95, nesterov=True,
                 whitening_prob=1.0, adamw_params=None, adamw_lr=3e-4, 
                 adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0):
        
        defaults = dict(
            lr=lr, lr_param=lr_param, momentum=momentum, nesterov=nesterov,
            whitening_prob=whitening_prob,
            adamw_lr_ratio=adamw_lr/lr if lr else 1.0,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps, adamw_wd=adamw_wd
        )
        
        params = list(muon_params)
        if adamw_params:
            params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Initialize parameter states
        for p in muon_params:
            state = self.state[p]
            state['use_muon'] = p.ndim >= 2 and p.numel() < 1e7  # Auto-detect matrix params
            
        if adamw_params:
            for p in adamw_params:
                self.state[p]['use_muon'] = False
                
        # Distributed training setup
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Process Muon-style parameters
            for p in filter(lambda p: self.state[p]['use_muon'], group['params']):
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                g = p.grad
                
                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                
                # Momentum update
                buf.mul_(group['momentum']).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=group['momentum'])
                else:
                    g = buf.clone()
                
                # Reshape for 2D processing
                original_shape = g.shape
                g_2d = g.view(original_shape[0], -1) if g.ndim > 2 else g
                m, n = g_2d.shape
                min_dim = min(m, n)
                
                # Initialize preconditioner state
                if 'Q' not in state:
                    state['Q'] = torch.eye(min_dim, device=g.device, dtype=g.dtype)
                if 'GhG' not in state:
                    state['GhG'] = torch.zeros((min_dim, min_dim), device=g.device, dtype=g.dtype)
                
                # Apply preconditioning
                precond_g, new_GhG, new_Q = muon_style_psgd(
                    g_2d, 
                    state['GhG'],
                    state['Q'],
                    lr_preconditioner=group['lr_param'],
                    preconditioner_update_probability=group['whitening_prob']
                )
                
                # Update state
                state['GhG'] = new_GhG
                state['Q'] = new_Q
                
                # Reshape back if needed
                if g.ndim > 2:
                    precond_g = precond_g.view(original_shape)
                
                # Parameter update
                p.add_(precond_g, alpha=-group['lr'])
            
            # Process AdamW fallback parameters
            for p in filter(lambda p: not self.state[p]['use_muon'], group['params']):
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                g = p.grad
                
                # Initialize AdamW state
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                state['step'] += 1
                beta1, beta2 = group['adamw_betas']
                
                # AdamW updates
                state['exp_avg'].mul_(beta1).add_(g, alpha=1-beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(g, g, value=1-beta2)
                
                denom = state['exp_avg_sq'].sqrt().add_(group['adamw_eps'])
                step_size = group['lr'] * group['adamw_lr_ratio']
                
                if group['adamw_wd'] > 0:
                    p.mul_(1 - step_size * group['adamw_wd'])
                
                bias_corr1 = 1 - beta1 ** state['step']
                bias_corr2 = 1 - beta2 ** state['step']
                step_size = step_size * (bias_corr2 ** 0.5) / bias_corr1
                
                p.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss
