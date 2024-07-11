import torch

normal_dist = torch.distributions.Normal(loc=0, scale=1.)


def score_function(X, y_clean, model, level=0, clamp_positive=False):
    """ Compute the score of sample(s).
        The score is the difference between the largest logit (except the one of the true class) and the logit of the true class.
        
        Args:
            X (tensor): input tensor of sample(s)
            y_clean (int or zero dimension tensor): true class
            model: pyTorch neural network
            level (float): relative level to compare the score
            clamp_positive (bool): remove (set to zero) the positive scores 
    
        Return:
            single value tensor of dimension 1
    """
    y_pred = model(X)
    y_diff = torch.cat((y_pred[:,:y_clean], y_pred[:,(y_clean + 1):]), dim=1) - y_pred[:,y_clean].unsqueeze(-1)
    score, _ = y_diff.max(dim=1)
    score -= level
    if clamp_positive:
        score = torch.clamp(score, max=0)
    return score


def get_model_accuracy(model, X, y):
    """ Compute the accuracy of the model on a supervised dataset (X,y)
    
        Args:
            model: pyTorch neural network
            X (tensor): input tensor of sample(s)
            y_clean (int or zero dimension tensor): true class
            
        Return:
            float: accuracy
    """
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.argmax(logits,-1)
        correct_idx = y_pred == y

    return correct_idx.float().mean()


class NormalCDFLayer(torch.nn.Module):
    """ Transformation layer for a uniform noise """

    def __init__(self, x_clean, epsilon=0.1, x_min=0., x_max=1., device='cpu'):
        super(NormalCDFLayer, self).__init__()
        self.x_clean = x_clean
        self.epsilon = epsilon
        self.x_min = x_min
        self.x_max = x_max
        self.device = device
        
    def forward(self,x):
        u = normal_dist.cdf(x).view(-1,*self.x_clean.shape)
        return torch.clip(self.x_clean + self.epsilon * (2 * u - 1), min=self.x_min, max=self.x_max)

    def inverse(self,x):
        return normal_dist.icdf(((x - self.x_clean) / self.epsilon + 1.) / 2.)

    def string(self):
        return f"NormalCDFLayer(epsilon={self.epsilon})"
    

class NormalToUnifLayer(torch.nn.Module):
    """ Transformation layer for a real uniform noise """

    def __init__(self, input_shape, low=0., high=1., device='cpu'):
        super(NormalToUnifLayer, self).__init__()
        self.input_shape = input_shape
        self.low = low
        self.high = high
        self.device = device
        self.range = self.high - self.low

    def forward(self, x):
        u = normal_dist.cdf(x).view(-1,*self.input_shape)
        return self.low + self.range * u

    def inverse(self, x):
        return normal_dist.icdf(((x - self.low) / self.range))

    def string(self):
        return f"NormalToUnifLayer(low={self.low}, high={self.high})"


class NormalClampReshapeLayer(torch.nn.Module):
    """ Transformation layer for a gaussian noise """

    def __init__(self, x_clean, x_min=0, x_max=1., sigma=1.):
        super(NormalClampReshapeLayer, self).__init__()
        self.x_clean = x_clean
        self.x_min = x_min
        self.x_max = x_max
        self.sigma = sigma

    def forward(self, x):
        x = x.view(-1,*self.x_clean.shape)
        return torch.clip(self.x_clean + self.sigma * x, min=self.x_min, max=self.x_max)        # LM: apply torch.clip on the final result instead of x

    def inverse(self,x): 
        return (x - self.x_clean) / self.sigma 

    def string(self):
        return f"NormalClampReshapeLayer(sigma={self.sigma})"
