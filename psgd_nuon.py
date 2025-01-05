import os
import torch
import torch.distributed as dist

def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def single_sided_whiteningn(G, Q, lr_precond=0.1):
    m, n = G.shape
    assert(m >= n)
    V = torch.randn_like(G)/m**0.5
    A = G @ Q.T
    Bh = torch.linalg.solve_triangular(Q, V, upper=True, left=False)
    AhA = A.T @ A
    BBh = Bh.T @ Bh
    Q = Q - lr_precond/torch.linalg.matrix_norm(AhA + BBh, ord=2) * torch.triu(AhA - BBh) @ Q
    return Q


class Nuon(torch.optim.Optimizer):
    """
    Nuon - MomentUm Orthogonalized by one sided psgd whitening

    Nuon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a learned one sided psgd whitening iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Arguments:
        nuon_params: The parameters to be optimized by Nuon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `nuon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, nuon_params, lr=0.02, lr_precond=0.1, momentum=0.95, nesterov=True, ns_steps=6,
                 adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0):

        defaults = dict(lr=lr, lr_precond=lr_precond, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        params = list(nuon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use Nuon, and those for which we will not
        for p in nuon_params:
            # Use Nuon for every parameter in nuon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_nuon'] = True
            else:
                self.state[p]['use_nuon'] = False
        for p in adamw_params:
            # Do not use Nuon for parameters in adamw_params
            self.state[p]['use_nuon'] = False

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

            
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Nuon           #
            ############################

            params = [p for p in group['params'] if self.state[p]['use_nuon']]
            lr = group['lr']
            momentum = group['momentum']

            # Process each parameter independently
            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]
                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Initialize or retrieve the Q matrix for this parameter
                if 'Q' not in state:
                    state['Q'] = torch.eye(g.shape[0], device=g.device)
                Q = state['Q']

                # Update Q matrix and apply gradient orthogonalization
                Q = single_sided_whiteningn(g, Q, lr_precond=group['lr_precond'])
                state['Q'] = Q
                g = g @ Q.T @ Q
                g *= max(1, g.size(0) / g.size(1))**0.5

                # Apply the gradient update
                p.data.add_(g, alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if not self.state[p]['use_nuon']]
            lr = group['adamw_lr_ratio'] * group['lr']  # Adjust learning rate
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]
                # Initialize AdamW state
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1

                # Compute AdamW update
                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']
                scale = bias_correction1 / bias_correction2**0.5

                # Apply weight decay and update parameters
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
