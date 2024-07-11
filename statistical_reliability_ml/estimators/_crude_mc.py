import torch

from statistical_reliability_ml._config import MethodConfig


class CrudeMC(MethodConfig):

    """ Crude Monte-Carlo estimation""" 

    name = "CrudeMC"
    is_weighted = False
    is_mpp = False
    is_parametric = False

    def __init__(self, N_samples=1000, batch_size=100, track_advs=False):
        """ Initialize the Crude Monte-Carlo experiment

            Args:
                N_samples (int): number of random samples for the MC experiment
                batch_size (int): batch_size to group samples during the analysis
                track_advs (bool): if True keep track of the adversarial examples found during the analysis
        """
        super().__init__()

        self.N_samples = N_samples
        self.batch_size = batch_size
        self.track_advs = track_advs

    def run_estimation(self, experiment=None, generator=None, score=None, verbose=0):
        """ Computes probability of failure 

        Args:
            experiment (StatsRelExperiment): experiment object 
            generator (func): random generator
            score (func): score function
            verbose (int): verbosity
           
        Returns:
            float: probability of failure
            dict: additional results:
                nb_calls: number of calls to the score function
                adv: adversarial examples found (if requested)
        """
        if experiment is None:
            assert (generator is not None) and (score is not None), 'CrudeMC requires a generator and a score function'
        else:
            generator = experiment.generator
            score = experiment.score

        self.N_samples = int(self.N_samples)
        if verbose > 2:
            print(f'Running CrudeMC estimation with {self.N_samples} samples')

        p_f = 0
        count_samples = 0
        x_advs = []
        
        # run by batch
        for _ in range(self.N_samples // self.batch_size):
            x_mc = generator(self.batch_size)
            h_MC = score(x_mc)
            if self.track_advs:
                x_advs.append(x_mc[h_MC >= 0])

            count_samples += self.batch_size
            p_f = ((count_samples - self.batch_size) / count_samples) * p_f + (self.batch_size / count_samples) * (h_MC >= 0).float().mean()

        # run the remaining steps    
        rest = self.N_samples % self.batch_size
        if rest != 0:
            x_mc = generator(rest)
            h_MC = score(x_mc)
            if self.track_advs:
                x_advs.append(x_mc[h_MC >= 0])
            count_samples += rest
            p_f = ((count_samples - rest) / count_samples) * p_f + (rest / count_samples) * (h_MC >= 0).float().mean()

        assert count_samples == self.N_samples
        
        dict_out = {'nb_calls': self.N_samples}
        if self.track_advs:
            dict_out['advs'] = torch.cat(x_advs, dim=0).to('cpu').numpy()
        
        return p_f.cpu(), dict_out
