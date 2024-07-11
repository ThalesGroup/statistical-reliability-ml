import scipy.stats as stats
import torch

from ._mpp_utils import MPPSearchConfig


class HLRF(MPPSearchConfig):
    
    """ Performs an attack to search the Most Probable Point using the Hasofer-Lind-Rackwitz-Flessler algorithm """

    name = 'HLRF'

    def __init__(self, max_iter=1000, stop_cond_type='grad_norm', stop_eps=1e-3, stepsize=0.2, mult_grad_calls=2., real_mpp=True, eps_real_mpp=1e-3):
        """ Configure the HLRF algorithm
    
            Args:
                max_iter (int): maximum number of iteration
                stop_cond_type (str): type of stop condition (either 'grad_norm', 'norm' or 'beta')\
                stop_eps (float): stop condition value
                stepsize (float): progression speed parameter
                mult_grad_calls (float): approximative multiplicative factor for gradient function calls vs score function calls  
                real_mpp (bool): if True find real MPP
                eps_real_mpp (float): stop condition for real MPP search
        """
        super().__init__(real_mpp=real_mpp, eps_real_mpp=eps_real_mpp)
        self.max_iter = max_iter
        self.stop_cond_type = stop_cond_type
        self.stop_eps = stop_eps
        self.stepsize = stepsize
        self.mult_grad_calls = mult_grad_calls

        if self.stop_cond_type not in ['grad_norm','norm','beta']:
            raise NotImplementedError(f"Method {self.stop_cond_type} is not implemented.")

    def _mpp_search(self, experiment=None, x_clean=None, gradient=None, debug=False, print_every=10):
        """ Run the HLRF algorithm
    
            Args:
                experiment (StatsRelExperiment): experiment object
                x_clean (Torch tensor): clean input
                gradient (func): gradient function                
                debug (bool): set debug mode
                print_every (int): if debug, print diagnosis at some iterations

            Return:
                tensor: MPP points
                int: number of calls to the score function
        """
        if experiment is None:
            assert x_clean is not None, 'HLRF requires the clean input'
            assert gradient is not None, 'HLRF requires the gradient function'
        else:
            x_clean = experiment.x_clean
            gradient = experiment.gradient

        neg_gradient = lambda x: [-1 * v for v in gradient(x)]
        x_current = torch.zeros((1, x_clean.numel()), device=x_clean.device)
        count_iter = 0
        stop_cond = False
        grad_x, score_x = neg_gradient(x_current)
        norm_grad = torch.norm(grad_x, dim=-1)
        alpha = -1 * grad_x / norm_grad
        beta = torch.sum(x_current * alpha, dim=1) + score_x / norm_grad
        grad_calls = 1

        while count_iter < self.max_iter and (not stop_cond or score_x > 0): 
            count_iter += 1
            x_new = x_current + self.stepsize * (beta * alpha - x_current)
            grad_xnew, score_x = neg_gradient(x_new)
            norm_grad = torch.norm(grad_xnew, dim=-1)
            alpha = -1 * grad_xnew / norm_grad
            beta_new = torch.sum(x_new * alpha, dim=1) + score_x / norm_grad
            grad_calls += 1

            if self.stop_cond_type == 'grad_norm':
                diff = torch.norm(grad_x - grad_xnew, dim=-1)

            elif self.stop_cond_type == 'norm':
                diff = torch.norm(x_current - x_new, dim=-1)

            elif self.stop_cond_type == 'beta':
                diff = torch.abs(beta - beta_new)
            
            stop_cond = (diff < self.stop_eps).item()
            beta = beta_new
            grad_x = grad_xnew
            x_current = x_new

            if debug and count_iter % print_every == 0:
                print(f"HLRF iteration {count_iter} / {self.max_iter}: {self.stop_cond_type}_diff: {float(diff)}")
                print(f'\tbeta: {float(beta)}')
                print(f'\tscore: {float(score_x)}')
                print(f"\tProb of failure: {float(stats.norm.cdf(-beta))}")

        if count_iter == self.max_iter:
            print("Warning: maximum number of iteration has been reached in HLRF")

        return x_current, self.mult_grad_calls * grad_calls
