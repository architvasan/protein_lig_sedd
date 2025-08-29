import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import protlig_dd.processing.graph_lib as graph_lib
from protlig_dd.model import utils as mutils


def get_loss_fn(noise, graph, train=True, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, perturbed_batch=None, t=None, prot_seq=None, lig_seq=None, task= "ligand_given_protein"):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps #t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        #print(f"{t=}")    
        sigma, dsigma = noise(t)
        #print(f"{sigma=}") 
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])
        print(f"{batch=}")
        print(f"{perturbed_batch=}")
        #print(f"{perturbed_batch=}")
        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma, prot_seq=prot_seq, lig_seq=lig_seq, task=task)#, esm_cond, mol_cond) # * 100
        #print(f"{log_score=}")
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)#/batch.shape[0]
        #print(loss)
        #print(f"{loss=}")
        #normalized_dsigma = dsigma / dsigma.mean()
        #safe_dsigma = dsigma.clamp(min=1e-6, max=1)#1.0)
        print(f"{dsigma=}")
        print(f"{loss=}")
        if True:
            loss = (dsigma[:, None] * loss).mean()#sum(dim=-1)
        #print(loss)
        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)
    #print(loss_fn)
    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, prot_seq=None, lig_seq=None, task= "ligand_given_protein"):
        #nonlocal accum_iter 
        #nonlocal total_loss
        
        accum_iter = 0
        total_loss = 0
        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, prot_seq=prot_seq, lig_seq=lig_seq, task= task).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn
