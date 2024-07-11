import scipy.stats as stats 

from statistical_reliability_ml._config import MethodConfig


class FORM(MethodConfig):
    
    name = 'FORM'
    is_weighted = False
    is_mpp = True
    is_parametric = False
                    
    def __init__(self, search_method, num_iter=100, save_mpp=False):
        """ Initialize the FORM experiment

            Args:
                search_method: search method for the Most Probable failure Points
                num_iter (int): number of iterations for the MPP search
                save_mpp (bool): save MPP in the results                
        """
        super().__init__()

        self.search_method = search_method
        self.num_iter = num_iter
        self.save_mpp = save_mpp
    
    def get_range_vars(self):
        range_vars = super().get_range_vars()
        search_method_vars = self.search_method.get_range_vars()
        return range_vars + search_method_vars

    def update_range_vars(self, list_params):
        """ update the parameters of the method and the search method
            Args:
                list_params: list with triplets each with method name, parameter name , and parameter value
        """
        super().update_range_vars(list_params)
        self.search_method.update_range_vars(list_params)

    def run_estimation(self, experiment=None, epsilon=0.2, x_mpp=None, verbose=0, **kwargs):
        """Computes the probability of failure using the First Order Method (FORM)

            Args:
                experiment (StatsRelExperiment): experiment object 
                epsilon (float): noise scale
                x_mpp: MPP points if already known
                verbose (int): Level of verbosity
                kwargs: additional arguments for MPP search methods

            Returns:
                p_fail (float): probability of failure
                dict_out (dict): dictionary containing the parameters of the attack
        """
        if experiment is not None:
            epsilon = experiment.epsilon

        if verbose > 2:
            print('Running FORM estimation')

        nb_calls = 0 
        if x_mpp is None:
            x_mpp, nb_calls = self.search_method.mpp_search(experiment, debug=verbose > 2, **kwargs)
            
        l2dist = x_mpp.norm(p=2).detach().cpu().item()
        p_fail = stats.norm.cdf(-l2dist / epsilon)
        dict_out = {'nb_calls': nb_calls, 'l2dist': l2dist}
        if self.save_mpp:
            dict_out['mpp'] = x_mpp.to('cpu').numpy()        
        
        return p_fail, dict_out
