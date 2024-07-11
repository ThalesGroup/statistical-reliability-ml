
import numpy as np
import scipy.stats as stats
import torch

from statistical_reliability_ml._config import MethodConfig


class ImportanceSampling(MethodConfig):
 
    name = 'ImportanceSampling'
    is_weighted = True
    is_mpp = True
    is_parametric = False
    
    def __init__(self, search_method, N_samples=int(1E4), batch_size=int(1E3), sigma_bias=1.,
                 save_mpp=False, save_weights=False, track_advs=False):
        """ Initialize the Importance Sampling experiment

            Args:
                search_method: search method for the Most Probable failure Points
                N_samples (int): number of random samples for the probability estimation
                batch_size (int): batch_size to group samples during the analysis
                sigma_bias (float):
                save_mpp (bool): save MPP in the results
                save_weights (bool): save bias weigths
                track_advs (bool): if True keep track of the adversarial examples found during the analysis
        """
        super().__init__()

        self.search_method = search_method
        self.N_samples = N_samples
        self.batch_size = batch_size
        self.sigma_bias = sigma_bias
        self.save_mpp = save_mpp
        self.save_weights = save_weights
        self.track_advs = track_advs

    def get_range_vars(self):
        range_vars = super(ImportanceSampling, self).get_range_vars()
        search_method_vars = self.search_method.get_range_vars()
        return range_vars + search_method_vars

    def update_range_vars(self, list_params):
        """ update the parameters of the method and the search method
            Args:
                list_params: list with triplets each with method name, parameter name , and parameter value
        """
        super(ImportanceSampling, self).update_range_vars(list_params)
        self.search_method.update_range_vars(list_params)

    def run_estimation(self, experiment=None, x_clean=None, generator=None, score=None, alpha_CI=0.05, x_mpp=None, verbose=0., **kwargs):
        """ Gaussian importance sampling algorithm to compute probability of failure

        Args:
            experiment (StatsRelExperiment): experiment object 
            x_clean (torch.Tensor): clean input
            generator (func):  random generator
            score (func): score function
            alpha_CI (float): confidence interval          
            x_mpp: MPP points if already known
            verbose (int): Level of verbosity
            kwargs: additional arguments for MPP search methods

        Returns:
            float: probability of failure
            dict: additional results
        """
        if experiment is None:
            assert x_clean is not None, 'ImportanceSampling requires the clean input'
            assert generator is not None, 'ImportanceSampling requires a random generator'
            assert score is not None, 'ImportanceSampling requires the score function'
        else:
            x_clean = experiment.x_clean
            generator = experiment.generator
            score = experiment.score
            alpha_CI = experiment.alpha_CI

        self.N_samples = int(self.N_samples)
        if verbose > 2:
            print(f'Running IS estimation with {self.N_samples} samples')

        nb_calls = 0 
        if x_mpp is None:
            x_mpp, nb_calls = self.search_method.mpp_search(experiment, debug=verbose > 2, **kwargs)
            
        dimension = x_clean.numel()
        zero_latent = torch.zeros((1, dimension), device=x_clean.device)
        x_mpp = x_mpp.reshape((1, dimension))
        beta_HL = torch.norm(x_mpp, dim=-1)
        if verbose > 2:
            print(f"beta_HL: {beta_HL}")

        gen_bias = lambda n: x_mpp + self.sigma_bias * generator(n)
        p_f = 0
        pre_var = 0
        count_samples = 0
        x_advs = []
        weights = []

        batch_size = min(self.batch_size, self.N_samples)
        for _ in range(self.N_samples // batch_size):
            x_mc = gen_bias(batch_size)
            with torch.no_grad():
                rare_event = score(x_mc) >= 0

            if self.track_advs:
                x_advs.append(x_mc[rare_event])
            
            count_samples += batch_size
            gauss_weights = GaussianImportanceWeight(x=x_mc, mu_1=zero_latent, mu_2=x_mpp, sigma_2=self.sigma_bias, d=dimension)
            p_local = (rare_event) * gauss_weights
            pre_var_local = (rare_event) * gauss_weights**2
            del x_mc, rare_event

            if self.save_weights:
                weights.append(p_local)

            p_f = ((count_samples - batch_size) / count_samples) * p_f + (batch_size / count_samples) * p_local.float().mean()
            pre_var = ((count_samples - batch_size) / count_samples) * pre_var + (batch_size / count_samples) * pre_var_local.float().mean()

        rest = self.N_samples % batch_size
        if rest != 0:            
            x_mc = gen_bias(rest)
            with torch.no_grad():
                rare_event = score(x_mc) >= 0
            if self.track_advs:
                x_advs.append(x_mc[rare_event])

            gauss_weights = GaussianImportanceWeight(x=x_mc, mu_1=zero_latent, mu_2=x_mpp, sigma_2=self.sigma_bias, d=dimension)
            
            p_local = (rare_event) * gauss_weights
            pre_var_local = (rare_event) * gauss_weights**2
            if self.save_weights:
                weights.append(p_local)

            del x_mc, rare_event
            count_samples += rest        
            p_f = ((count_samples - rest) / count_samples) * p_f + (rest / count_samples) * p_local.float().mean()
            pre_var = ((count_samples - rest) / count_samples) * pre_var + (rest / count_samples) * pre_var_local.float().mean()
        
        dict_out = {'nb_calls': count_samples + nb_calls}
        var_est = (1 / count_samples) * (pre_var - p_f**2).item()
        dict_out['var_est'] = var_est
        dict_out['std_est'] = np.sqrt(var_est)
        dict_out['CI'] = stats.norm.interval(1 - alpha_CI, loc=p_f.item(), scale=dict_out['std_est'])
        if self.save_weights:
            weights = torch.cat(weights, dim=0).reshape((1, self.N))
            dict_out['weights'] = weights.to('cpu').numpy()
        if self.save_mpp:
            dict_out['mpp'] = x_mpp.to('cpu').numpy()        
        if self.track_advs:
            dict_out['advs'] = torch.cat(x_advs, dim=0).to('cpu').numpy()

        return p_f.cpu().item(), dict_out


def GaussianImportanceWeight(x, mu_1, mu_2, sigma_1=1., sigma_2=1., d=1):
    """ Computes importance weights for Gaussian distributions for Importance Sampling

    Args:

        x (torch.tensor): input
        mu_1 (torch.tensor): mean of first Gaussian
        mu_2 (torch.tensor): mean of second Gaussian
        sigma_1 (torch.tensor): standard deviation of first Gaussian
        sigma_2 (torch.tensor): standard deviation of second Gaussian

    Returns:
        torch.tensor: importance weights i.e. ratio of densities of the two Gaussians at x
    """
    return torch.exp(-0.5 * ((x - mu_1) / sigma_1).square().sum(-1) + 0.5 * ((x - mu_2) / sigma_2).square().sum(-1)) * (sigma_2**d / sigma_1**d)
    
