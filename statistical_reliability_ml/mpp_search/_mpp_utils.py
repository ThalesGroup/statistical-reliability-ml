from statistical_reliability_ml._config import MethodConfig


def _binary_search_to_zero(function, x_init, lambda_min=0., lambda_max=4., eps=1e-3, max_iter=32, verbose=False):
    """ Binary search to find the zero of a function """
    i = 1
    if function(x_init) > 0:
        a = 1
        b = lambda_max
    else:
        a = lambda_min
        b = 1

    c = (a + b) / 2
    f_c = function(c * x_init)

    while f_c.abs() > eps and i < max_iter:
        i += 1
        if verbose:
            print(f"c={c}, f_c={f_c}")

        if f_c > 0:
            a = c
        else:
            b = c
        c = (a + b) / 2
        f_c = function(c * x_init)
    
    if verbose and i == max_iter:
        print("Warning: maximum number of iteration has been reached in binary search")
    return c, i


class MPPSearchConfig(MethodConfig):
    """ Base class for Most Probable failure Points search method"""

    name = 'mppsearch'

    def __init__(self, real_mpp=True, eps_real_mpp=1e-3, real_mpp_max_iter=32, lambda_min=0, lambda_max=4):
        """ Configure the search of real MPP points
            Args:
                real_mpp (bool): if True find real MPP
                eps_real_mpp (float): stop condition for real MPP search
                real_mpp_max_iter (int): maximum iteration of the binary search for real MPP
                lambda_min (float):
                lambda_max (float):
        """
        self.real_mpp = real_mpp
        self.eps_real_mpp = eps_real_mpp
        self.real_mpp_max_iter = real_mpp_max_iter
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
    
    def _mpp_search(self, experiment, **kwargs):
        raise Exception('Not implemented')
        
    def mpp_search(self, experiment=None, score=None, debug=False, **kwargs):
        """ Search MPP and then find real MPP
            
            Args:
                experiment (StatsRelExperiment): experiment object
                score (func): score function
                debug (bool): debug mode
                kwargs: arguements for the MPP search
        """
        x_mpp, nb_calls = self._mpp_search(experiment, debug=debug, **kwargs)
        
        if debug:
            print(f'MPP search performed with {nb_calls} calls to the score function')
        
        if self.real_mpp:
            if experiment is None:
                assert score is not None, 'True MPP search requires the score function'
            else:
                score = experiment.score
            neg_score = lambda x: -1 * score(x)

            lambda_, nb_calls_ = _binary_search_to_zero(neg_score, x_mpp, lambda_min=self.lambda_min, lambda_max=self.lambda_max,
                                                        eps=self.eps_real_mpp, max_iter=self.real_mpp_max_iter)   
            x_mpp = lambda_ * x_mpp
            nb_calls += nb_calls_

            if debug:
                print(f'Real MPP search performed with {nb_calls_} calls to the score function')
        
        return x_mpp, nb_calls
