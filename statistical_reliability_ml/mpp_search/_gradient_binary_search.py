import torch

from ._mpp_utils import MPPSearchConfig


class GradientBinarySearch(MPPSearchConfig):
    
    """ Search the Most Probable Point using a simple binary search  in the direction of the gradient"""
    
    name = 'GradientBinarySearch'

    def __init__(self, num_iter=20, max_iter=100, alpha=0.5, mult_grad_calls=2., real_mpp=True, eps_real_mpp=1e-3):
        """ Configure the binary search
            
            Args:
                num_iter (int): number of iterations
                max_iter (int): maximum iterations for gradient descent
                alpha (float): speed factor
                real_mpp (bool): if True find real MPP
                eps_real_mpp (float): stop condition for real MPP search
        """
        super().__init__(real_mpp=real_mpp, eps_real_mpp=eps_real_mpp)
        self.num_iter = num_iter
        self.max_iter = max_iter
        self.alpha = alpha
        self.mult_grad_calls = mult_grad_calls

    def _mpp_search(self, experiment=None, x_clean=None, score=None, gradient=None, debug=False):
        """ Binary search algorithm to find the failure point in the direction of the gradient of the limit state function.

        Args:
            experiment (StatsRelExperiment): experiment object 
            x_clean (Torch tensor): clean input
            score (func): score function
            gradient (func): gradient function
            debug (bool): debug mode

        Returns:
            torch.tensor: failure point
            int: number of calls to the score function
        """
        if experiment is None:
            assert x_clean is not None, 'GradientBinarySearch requires the clean input'
            assert score is not None, 'GradientBinarySearch requires the score function'
            assert gradient is not None, 'GradientBinarySearch requires the gradient function'
        else:
            x_clean = experiment.x_clean
            score = experiment.score
            gradient = experiment.gradient

        neg_score = lambda x: -1 * score(x)
        neg_gradient = lambda x: [-1 * v for v in gradient(x)]
        x_current = torch.zeros((1, x_clean.numel()), device=x_clean.device)
        grad_x, score_x = neg_gradient(x_current)
        nb_calls = self.mult_grad_calls
        count_iter = 0

        while score_x > 0 and count_iter < self.max_iter:
            count_iter += 1
            x_current += self.alpha * grad_x
            grad_x, score_x = neg_gradient(x_current)
            nb_calls += self.mult_grad_calls
        
        if debug and count_iter == self.max_iter:
            print("Warning: maximum number of iteration has been reached in GradientBinarySearch")

        a = torch.zeros((1, x_clean.numel()), device=x_clean.device)
        b = x_current 
        for _ in range(self.num_iter):
            c = (a + b) / 2
            score_c = neg_score(c) * -1
            nb_calls += 1

            if score_c > 0:
                b = c
            else:
                a = c
        
        return b, nb_calls
