from datetime import datetime
from itertools import product as cartesian_product
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy.stats as stats
from time import time
from tqdm.auto import tqdm
import torch

from statistical_reliability_ml._config import ExperimentConfig
import statistical_reliability_ml._torch_utils as torch_utils
import statistical_reliability_ml._utils as utils


class StatsRelExperiment(ExperimentConfig):

    name = 'StatsRelExperiment'
    
    def __init__(self, model, X, y,
                 results_file=None, 
                 results_dir=None,
                 n_rep=100,
                 nb_points_errorgraph=100,
                 epsilon_range=[],
                 eps_min=0.2,
                 eps_max=0.3,
                 eps_num=5,
                 noise_dist='uniform',
                 x_min=0,
                 x_max=1,
                 low=None,
                 high=None,                 
                 mask_cond=None,
                 mask_idx=None,
                 level=0,
                 clamp_positive=False,
                 p_ref=None,
                 alpha_CI=0.05,        
                 **kwargs):
        """ Configure the experiments
        
            Args:
                model (pyTorch model): neural network model
                X (pyTorch tensor): dataset of samples to analyze
                y (pyTorch tensor): classes of the samples
                results_file (str or Path): path of the CSV file use to save results. 
                    If the file exists results are appended to the file. If None no result is saved.
                results_dir (str or Path): path of a directory were results of each experiment are saved.
                n_rep (int): number of repetitions of each experiment
                nb_points_errorgraph (int): number of points to plot in the error graph. If 0 no graph is generated.
                epsilon_range (list): list of epsilon values to test. If empty a range of eps_num values is drawn from eps_min to eps_max.
                eps_min (float): min value for epsilon  
                eps_max (float): max value for epsilon
                eps_num (int): number of values for epsilon
                noise_dist (str): noise distribution (eitehr uniform, real_uniform or gaussian)
                x_min (float): min values of the input samples
                x_max (float): max values of the input samples
                low (float): lower values for the input sample with noise. Usually set to None to compute the value according to the current sample. 
                high (float): higher values for the input sample with noise. Usually set to None to compute the value according to the current sample.
                mask_cond (array): mask input samples according to a cond, such that no perturbation is applied if cond is satisfied.
                mask_idx (array): mask input samples according to indices, such that no perturbation is applied on the masked indices.
                level (float): score level to define adversarial examples (usually 0)
                clamp_positive (bool): clip positive value of the score.
                p_ref (float): reference probability to compare the estimation
                alpha_CI (float): confidence interval for the estimation                 
        """
        super().__init__(**kwargs)

        self.model = model
        self.X = X
        self.y = y
        self.input_shape = self.X.shape[1:]
        self.input_dimension = np.prod(self.input_shape)
        self.results_file = results_file
        self.results_dir = results_dir
        self.n_rep = n_rep
        if self.n_rep <= 0:
            raise ValueError('The number of repetitions must be at least 1')
        self.nb_points_errorgraph = nb_points_errorgraph
        self.epsilon_range = epsilon_range
        if len(self.epsilon_range) == 0:
            # get eps_num epsilon values between eps_min and eps_max (with a logarithmic progression)
            log_line = np.linspace(start=np.log(eps_min), stop=np.log(eps_max), num=eps_num)
            self.epsilon_range = np.exp(log_line)
        self.noise_dist = noise_dist        
        if self.noise_dist not in ['uniform', 'real_uniform', 'gaussian']:
            raise NotImplementedError("Only uniform, real_uniform and gaussian distributions are implemented.")        
        self.x_min = x_min
        if not isinstance(self.x_min, torch.Tensor):
            self.x_min = self.x_min * torch.ones(self.input_shape)
        self.x_max = x_max
        if not isinstance(self.x_max, torch.Tensor):
            self.x_max = self.x_max * torch.ones(self.input_shape)
        self.low = low
        self.high = high        
        self.mask_cond = mask_cond
        self.mask_idx = mask_idx
        self.level = level
        self.clamp_positive = clamp_positive
        self.p_ref = p_ref
        self.alpha_CI = alpha_CI
        self.q_alpha_CI = stats.norm.ppf(1 - self.alpha_CI / 2)
        
        self.x_clean = None
        self.y_clean = None
        self.current_low = None
        self.current_high = None
        self.epsilon = None
          
    def update(self, input_index : int, epsilon : float):
        """ Update the experiment with the index of the sample and the value of epsilon.
            Update the attributes current_low, current_high and t_transform

            Args:
                input_index (int): index of the input to test (in the dataset X)
                epsilon (float): value of epsilon to test
        """
        self.x_clean = self.X[input_index]
        self.y_clean = self.y[input_index]
        self.epsilon = epsilon

        if self.low is None:
            self.current_low = torch.maximum(self.x_clean - epsilon, self.x_min.view(-1,*self.input_shape).to(self.device))
        elif not isinstance(self.low, torch.Tensor):
            self.current_low = self.low * torch.ones(self.input_shape)
        else:    
            self.current_low = self.low

        if self.high is None:
            self.current_high = torch.minimum(self.x_clean + epsilon, self.x_max.view(-1,*self.input_shape).to(self.device))
        elif not isinstance(self.high, torch.Tensor):
            self.current_high = self.high * torch.ones(self.input_shape)
        else:    
            self.current_high = self.high

        if self.mask_cond is not None:
            idx = self.mask_cond(self.x_clean)
            self.current_low[idx] = self.x_clean[idx]
            self.current_high[idx] = self.x_clean[idx]
        elif self.mask_idx is not None:
            self.current_low[self.mask_idx] = self.x_clean[self.mask_idx]
            self.current_high[self.mask_idx] = self.x_clean[self.mask_idx]
        
        if self.noise_dist == 'uniform':
            self.t_transform = torch_utils.NormalCDFLayer(x_clean=self.x_clean, epsilon=epsilon, x_min=self.x_min, x_max=self.x_max, device=self.device)

        elif self.noise_dist == 'real_uniform':
            self.t_transform = torch_utils.NormalToUnifLayer(input_shape=self.input_shape, low=self.current_low, high=self.current_high, device=self.device)

        elif self.noise_dist in ['gaussian', 'normal']:
            self.t_transform = torch_utils.NormalClampReshapeLayer(x_clean=self.x_clean, x_min=self.x_min, x_max=self.x_max, sigma=self.epsilon)
            
    def generator(self, N : int):
        """ Generate random samples in the input dimension using a standard Gaussian distribution

            Args:
                int N: number of samples to generate
            Return:
                Torch tensor oof size (N, self.input_dimension)
        """
        return torch.randn(size=(N,self.input_dimension), device=self.device)        

    def score(self, X):
        """ Compute the score the samples """
        with torch.no_grad(): 
            x_u = self.t_transform(X)
            return torch_utils.score_function(x_u, self.y_clean, self.model, self.level, self.clamp_positive)
        
    def gradient(self, X):
        """ Compute the gradient and the score of the samples """
        X.requires_grad = True
        x_u = self.t_transform(X)
        score = torch_utils.score_function(x_u, self.y_clean, self.model, self.level, self.clamp_positive)
        gradient = torch.autograd.grad(outputs=score, inputs=X, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
        X.requires_grad = False
        gradient = gradient.view(X.shape)
        return gradient.detach(), score.detach()
    
    def run_estimation(self, method, test_indices=[], verbose=0, previous_results=None, **kwargs):
        """ Running statistical reliability experiments on a neural network model with supervised data (X,y) values
            Run one experiment for each indices, each epsilon value and each parameters configuration of the method.
            - Repeat each experiment if attribute n_rep > 1
            - Save results in a CSV file and a directory (if asked).
            - Plot results graph (if asked).
            
            Args:
                method (MethodConfig): estimation method to use
                test_indices (list): list of indices to analyze in the dataset self.X
                verbose (int): verbosity level
                previous_results (pd.DataFrame): dataframe  with previous results to concatenate  with the new results
                kwargs: additional arguments for the estimation method

            Return:
                pd.DataFrame: DataFrame with all the results (one line per experiment)
                dict: additional results
        """
        method_name = method.name
              
        if not test_indices:
            # testing all samples in X
            test_indices = list(range(len(self.X)))
            if verbose:
                print(f'Testing all samples in test dataset ({len(test_indices)} samples)')
        
        # compute the accuracy of the model on the dataset """
        if verbose:
            sample_accuracy = torch_utils.get_model_accuracy(self.model, self.X, self.y)
            print(f"Model accuracy on dataset: {sample_accuracy}")
        
        # compute the number of experiments as the product of parameters values, number of epsilon values and number of test indices
        method_range_params = method.get_range_vars()
        range_params_list = [[(m, k, v) for v in list_values] for m, k, list_values in method_range_params]
        list_params = list(cartesian_product(*range_params_list))
        nb_exps = len(list_params) * len(self.epsilon_range) * len(test_indices)

        if verbose > 0:
            print(f"Running {nb_exps} reliability experiments with method {method_name} on indices {test_indices}.")
            print(f"Testing range parameters: {method_range_params}")
            print(f"Testing {self.noise_dist} noise pertubation with epsilon in {self.epsilon_range}")
            if self.n_rep > 1:
                print(f"Each experiment is repeated {self.n_rep} times")
        if verbose > 1:
            print(self)
            print(method)

        # running experiments
        output_dict = {}
        results_list = []
        i_exp = 0
        with tqdm(total=nb_exps, disable=(verbose == 0)) as pbar:
            for count_input, input_index in enumerate(test_indices):
                for count_epsilon, epsilon in enumerate(self.epsilon_range):
                    for method_params in list_params:
                        i_exp += 1
                        pbar.update(1)
                        if verbose > 1:
                            print("-------------------------------------------------------------------------------")
                            print(f"Starting {method_name} estimation {i_exp}/{nb_exps}")
                            print(f"Using input_idx: {input_index} with epsilon: {epsilon} and parameters: {method_params}")
                        
                        # update configurations
                        self.update(input_index, epsilon)
                        method.update_range_vars(method_params)
                        
                        # configure output path
                        experiment_path = None
                        if self.results_dir is not None:
                            datetime_str = datetime.today().strftime('%Y-%m-%d_%Hh%Mm%S_%f')
                            experiment_path = Path(self.results_dir, method_name, datetime_str)
                            os.makedirs(experiment_path, exist_ok=False)
                            if verbose > 1:
                                print(f'Saving results of this experiment in {experiment_path}')

                        run_times = []
                        nb_calls = []
                        estimations = [] 
                        log_estimations = []
                        var_estimations = []

                        # run the experiment several times                    
                        for rep in tqdm(range(self.n_rep), disable=(verbose <= 1)):
                            start_time = time()

                            # run reliability estimation
                            p_est, dict_out = method.run_estimation(self, **kwargs, verbose=verbose)
                            
                            if verbose > 2:
                                print(f"Repetition {rep+1}/{self.n_rep}: P_estimation={p_est}")
                            
                            # collect results
                            run_times.append(time() - start_time)
                            nb_calls.append(dict_out['nb_calls'])
                            estimations.append(p_est)
                            log_estimations.append(np.log(p_est) if p_est > 0 else -250.)
                            
                            if 'var_est' in dict_out:
                                var_estimations.append(dict_out['var_est'])
                            elif 'std_est' in dict_out:
                                var_estimations.append(dict_out['std_est']**2)
                            
                            if 'advs' in dict_out:
                                if 'advs_list' not in output_dict:
                                    output_dict['advs_list'] = []
                                output_dict['advs_list'].append(dict_out['advs'])
                                
                            if 'mpp' in dict_out:
                                if 'mpp_list' not in output_dict:
                                    output_dict['mpp_list'] = []
                                output_dict['mpp_list'].append(dict_out['mpp'])

                            if 'weights' in dict_out:
                                if 'weights_list' not in output_dict:
                                    output_dict['weights_list'] = []
                                output_dict['weights_list'].append(dict_out['weights'])

                        # aggregate results for this experiment
                        run_times = np.array(run_times)
                        nb_calls = np.array(nb_calls)
                        estimations = np.array(estimations)
                        log_estimations = np.array(log_estimations)                
                        var_estimations = np.array(var_estimations)
                        mean_estimation = estimations.mean()
                        std_estimation = estimations.std()
                        mean_nb_calls = nb_calls.mean()
                        q1_estimation, med_estimation, q3_estimation = np.quantile(a=estimations, q=[0.25,0.5,0.75])
                        log_q1_estimation, log_med_estimation, log_q3_estimation = np.quantile(a=log_estimations, q=[0.25,0.5,0.75])
                        std_rel = std_estimation / mean_estimation**2 if mean_estimation > 0 else 0
                        
                        result = {'method_name': method_name,
                                  'input_index': input_index,
                                  'mean_calls': mean_nb_calls,
                                  'std_calls': nb_calls.std(),
                                  'mean_time': run_times.mean(), 
                                  'std_time': run_times.std(),
                                  'mean_estimation': mean_estimation,
                                  'std_estimation': std_estimation,
                                  'std_rel': std_rel, 
                                  'std_rel_adj': std_rel * mean_nb_calls,
                                  'q1_estimation': q1_estimation,
                                  'med_estimation': med_estimation,
                                  'q3_estimation': q3_estimation,
                                  'mean_log_estimation': log_estimations.mean(),
                                  'std_log_estimation': log_estimations.std(),
                                  'log_q1_estimation': log_q1_estimation,
                                  'log_med_estimation': log_med_estimation,
                                  'log_q3_estimation': log_q3_estimation }

                        if experiment_path is not None:
                            result['experiment_path'] = str(experiment_path)
                        else:
                            result['experiment_path'] = ''

                        if verbose > 1:
                            print(f"mean estimation: {mean_estimation}, std est:{std_estimation}, mean calls: {mean_nb_calls}")
                            print(f"std. rel.: {std_rel}, std. rel. adj.: {result['std_rel_adj']}")
                            
                        if self.p_ref is not None:
                            rel_error = np.abs(estimations - self.p_ref) / self.p_ref
                            result['p_ref'] = self.p_ref
                            result['rel_error'] = rel_error.mean()
                            result['std_rel_error'] = rel_error.std()

                            if verbose:
                                print(f"mean rel. error: {rel_error.mean()}")
                                print(f"std rel. error: {rel_error.std()}")
                                print(f"stat performance (per 1k calls) :{rel_error.std() * (mean_nb_calls / 1000)}")
                            
                        if experiment_path is not None:
                            # log arrays
                            np.savetxt(fname=Path(experiment_path, 'times.txt'), X=run_times)
                            np.savetxt(fname=Path(experiment_path, 'estimations.txt'), X=estimations)
                            np.savetxt(fname=Path(experiment_path, 'log_estimations.txt'), X=log_estimations)

                            # log histograms
                            plt.hist(run_times, bins=10)
                            plt.xlabel('time (s)')
                            plt.ylabel('nb repetitions')
                            plt.title('Histogram with the execution times for each repetition')
                            plt.savefig(Path(experiment_path, 'times_hist.png'))
                            plt.close()
                            plt.hist(estimations,bins=10)
                            plt.xlabel('probability')
                            plt.ylabel('nb repetitions')
                            plt.title('Histogram with the probability failure for each repetition')
                            plt.savefig(Path(experiment_path, 'estimations_hist.png'))
                            plt.close()
                            plt.hist(log_estimations, bins=10)
                            plt.xlabel('log probability')
                            plt.ylabel('nb repetitions')
                            plt.title('Histogram with the logarithmic probability failure for each repetition')
                            plt.savefig(Path(experiment_path, 'log_estimations_hist.png'))
                            plt.close()
                        
                        if self.nb_points_errorgraph > 0:
                            # plot an error graph
                            plot_title = fr'Estimation of failure probability with {method_name} for input idx.={input_index} and $\varepsilon$={self.epsilon:.3f}'
                            plot_title += fr'using $\approx${int(mean_nb_calls)} model calls with {self.n_rep} repetitions'
                            self.plot_errorgraph(estimations, var_estimations, plot_title, experiment_path, show=verbose > 1)

                        result.update(utils.simple_vars(method))
                        result.update(utils.simple_vars(self))
                        results_list.append(result)

        new_results = pd.DataFrame(results_list)
        if previous_results is not None:
            new_results = pd.concat([previous_results, new_results], ignore_index=True)

        # restore the range parameters of the method
        method.update_range_vars(method_range_params)

        # save results in CSV
        if self.results_file is not None:
            results_file = Path(self.results_file)
            if results_file.exists():
                # append to existing CSV file
                previous_results = pd.read_csv(results_file)
                new_results = pd.concat([previous_results, new_results], ignore_index=True)
            new_results.to_csv(self.results_file, index=False)

        if 'advs_list' in output_dict:
            output_dict['advs_list'] = np.concatenate(output_dict['advs_list'])
        if 'mpp_list' in output_dict:
            output_dict['mpp_list'] = np.concatenate(output_dict['mpp_list'])
        if 'weights_list' in output_dict:
            output_dict['weights_list'] = np.concatenate(output_dict['weights_list'])

        return new_results, output_dict

    def plot_errorgraph(self, estimations, var_estimations, plot_title='', experiment_path=None, show=True):
        """Plot a graphic with the probability estimations.
           Args:
                estimations: array with probability estimations
                var_estimations: array with estimated variance of each estimation (can be empty)
                plot_title: string witht the title of the plot
                experiment_path: path of the experiment to save the fig (can be None)
                show: show the plot on the console / notebook
        """
        nb_points = min(self.nb_points_errorgraph, self.n_rep)
        errorbar_idx = np.arange(nb_points)
        mean_estimation = estimations.mean()
        std_estimation = estimations.std()
        estimations = estimations[errorbar_idx]

        plt.figure(figsize=(13,8))
        if len(var_estimations) > 0:
            std_estimations = np.sqrt(var_estimations)[errorbar_idx]    
            plt.errorbar(x=errorbar_idx, y=estimations, yerr=self.q_alpha_CI * std_estimations, fmt='o', errorevery=1, elinewidth=1., capsize=2,
                         label = r'Est. Prob. +/- $q_{\alpha/2}\hat{\sigma}$')
        else:
            plt.errorbar(x=errorbar_idx, y=estimations, fmt='o', errorevery=1, elinewidth=1. ,capsize=2, 
                         label = r'Est. Prob. +/- $q_{\alpha/2}\hat{\sigma}$')

        if self.p_ref is not None:
            plt.plot(errorbar_idx, np.ones(nb_points) * self.p_ref, 'r', label=f'Ref. Prob. = {self.p_ref:.2E} ')

        plt.plot(errorbar_idx, np.ones(nb_points) * mean_estimation, 'g', label=f'Avg. Est.= {mean_estimation:.2E}, Std. Est. = {std_estimation:.2E}')
        CI_low_empi = mean_estimation - self.q_alpha_CI * std_estimation
        CI_high_empi = mean_estimation + self.q_alpha_CI * std_estimation
        plt.plot(errorbar_idx, np.ones(nb_points) * CI_low_empi, '--', color='grey', label=fr'CI low = {CI_low_empi:.2E}, $\alpha$ = {self.alpha_CI}')
        plt.plot(errorbar_idx, np.ones(nb_points) * CI_high_empi, '--', color='k', label=fr'CI high = {CI_high_empi:.2E}, $\alpha$ = {self.alpha_CI}')
        plt.xlabel('replicate')
        plt.ylabel('estimated probability')
        plt.legend(loc='upper right')
        plt.title(plot_title)

        if experiment_path is not None:
            plt.savefig(Path(experiment_path, 'errorbar.png'))
        if show:
            plt.show()
        plt.close()
