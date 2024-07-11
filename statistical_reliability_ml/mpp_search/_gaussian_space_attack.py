from math import sqrt
import torch

from ._mpp_utils import MPPSearchConfig


foolbox_attack_list = ['carliniwagner', 'carlini', 'carlini-wagner', 'cw', 'carlini_wagner', 'carlini_wagner_l2', 'adv', 'adv_attack',
                       'brendel', 'brendel-bethge', 'brendel_bethge', 'brendelbethge', 
                       'fmna','fmna_attack','fmna_attack_l2']


class GaussianSpaceAttack(MPPSearchConfig):
    
    """ Performs an attack on the latent space of the model using the Foolbox library """

    name = 'GaussianSpaceAttack'
    requires_expe = ['x_clean', 'y_clean', 'model', 't_transform', 'noise_dist']

    def __init__(self, attack='Carlini', num_iter=10, steps=100, stepsize=1e-2, max_dist=None, sigma=1., 
                 x_min=-int(1e2), x_max=int(1e2), random_init=False , sigma_init=0.5, real_mpp=True, eps_real_mpp=1e-3):
        """ Configure the attack
        
            Args:
                attack : name of the adersarial attack to use or an already configured Foolbox attack object
                num_iter (int): number of iteration for the binary search
                steps (int): number of steps for the attack
                stepsize (float): size of the steps
                max_dist (float): maximum distorsion
                sigma (float): distorsion factor if max_dist is None
                x_min (int): lower bounds of the perturbed samples
                x_max (int): upper bounds of the perturbed samples
                random_init (bool): initialize the search with random samples
                sigma_init (float): multiplicative factor for random initial samples
                real_mpp (bool): if True find real MPP
                eps_real_mpp (float): stop condition for real MPP search
        """
        super().__init__(real_mpp=real_mpp, eps_real_mpp=eps_real_mpp)
        self.attack = attack
        self.num_iter = num_iter
        self.steps = steps
        self.stepsize = stepsize
        self.max_dist = max_dist
        self.sigma = sigma
        self.x_min = x_min
        self.x_max = x_max
        self.random_init = random_init
        self.sigma_init = sigma_init

    def _mpp_search(self, experiment=None, x_clean=None, y_clean=None, model=None, t_transform=None, noise_dist='uniform', debug=False, **kwargs):
        """ Performs an attack on the latent space of the model using the Foolbox library
        
            Args:
                experiment (StatsRelExperiment): experiment object
                x_clean (torch.Tensor): clean input
                y_clean (torch.Tensor): clean label
                model (torch.nn.Module): model to attack
                t_transform: U-space transformation
                noise_dist (str): distribution of the noise
                debug (bool): debug mode
                kwargs: additional arguments for the attack
                
            Return:
                tensor: MPP points
                int: 0 value as the number of calls is unknown
        """       
        import foolbox

        if experiment is None:
            assert x_clean is not None, 'GaussianSpaceAttack requires the clean input'
            assert y_clean is not None, 'GaussianSpaceAttack requires the clean target class'
            assert model is not None, 'GaussianSpaceAttack requires the model'
        else:
            x_clean = experiment.x_clean
            y_clean = experiment.y_clean
            model = experiment.model
            t_transform = experiment.t_transform
            noise_dist = experiment.noise_dist

        device = x_clean.device
        max_dist = self.max_dist
        if max_dist is None:
            max_dist = sqrt(x_clean.numel()) * self.sigma
    
        if type(self.attack) is str:
            if self.attack.lower() in ('carlini', 'cw', 'carlini-wagner', 'carliniwagner', 'carlini_wagner', 'carlini_wagner_l2'):
                attack = foolbox.attacks.L2CarliniWagnerAttack(binary_search_steps=self.num_iter, stepsize=self.stepsize, steps=self.steps, **kwargs)

            elif self.attack.lower() in ('brendel', 'brendel-bethge', 'brendel_bethge'):
                attack = foolbox.attacks.L2BrendelBethgeAttack(binary_search_steps=self.num_iter, lr=self.stepsize, steps=self.steps, **kwargs)

            elif self.attack.lower() in ('fmna','fast_mininum_norm_attack','fmna_l2'):
                attack = foolbox.attacks.L2FMNAttack(binary_search_steps=self.num_iter, gamma=self.stepsize, steps=self.steps, **kwargs)

            else:
                raise NotImplementedError(f"Search method '{attack}' is not implemented.")
        
        if noise_dist.lower() in ('gaussian', 'normal'):
            fmodel = foolbox.models.PyTorchModel(model, bounds=(self.x_min, self.x_max), device=device)
            
            x_0 = x_clean            
            if self.random_init:
                x_0 += self.sigma_init * torch.randn_like(x_clean)
            
            _, advs, success = attack(fmodel, x_0.unsqueeze(0), y_clean.unsqueeze(0), epsilons=max_dist)
            assert success.item(), "The attack failed. Try to increase the number of iterations or steps."
            design_point = advs[0] - x_clean
            
        else:
            assert (t_transform is not None), "t_transform must be provided for a uniform noise"
            total_model = torch.nn.Sequential(t_transform, model)
            if self.random_init:
                x_0 = self.sigma_init * torch.randn_like(x_clean)
            else:
                x_0 = torch.zeros_like(x_clean)

            total_model.eval()
            fmodel = foolbox.models.PyTorchModel(total_model, bounds=(self.x_min, self.x_max), device=device)
            _, advs, success = attack(fmodel, x_0.unsqueeze(0), y_clean.unsqueeze(0), epsilons=max_dist)
            assert success.item(), "The attack failed. Try to increase the number of iterations or steps."
            design_point = advs[0]

        return design_point, 0
