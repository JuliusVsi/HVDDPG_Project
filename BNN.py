import torch
import torch.nn.functional as fun
from torch.nn import Parameter
import math
import numpy as np


def pho_to_std(pho):
    if isinstance(pho, float):
        return math.log(1 + math.exp(pho))
    return (1 + pho.exp()).log()


def std_to_pho(std):
    if isinstance(std, float):
        return math.log(math.exp(std) - 1)
    return (std.exp() - 1).log()


def nla_relu(inplace=True):
    def nla():
        return torch.nn.ReLU(inplace)
    return nla()


# KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian
def kl_div_p_q(p_mean, p_std, q_mean, q_std):
    if isinstance(p_std, float):
        p_std = torch.tensor(p_std, dtype=torch.float32)
    if isinstance(q_std, float):
        q_std = torch.tensor(q_std, dtype=torch.float32)
    numerator = (p_mean - q_mean)**2 + (p_std**2) - (p_std**2)
    denominator = 2 * (q_std**2) + 1e-8
    result = torch.sum(numerator / denominator + torch.log(q_std) - torch.log(p_std))

    return result


# ????
def log_prob_normal(inputs, mu=0., sigma=1.):
    log_normal = - math.log(sigma) - math.log(math.sqrt(2 * np.pi)) - \
                 (inputs - mu)**2 / (2 * sigma**2)
    result = torch.sum(log_normal)

    return result


###############################################################
# Likelihood Criterion
# Comment:predictions: (N, output_num) float
# targets: (N, output_num) float
# return value: scalar
###############################################################
def likelihood_criterion(predictions, targets, likelihood_sd=5.0):
    assert len(predictions) == len(targets)
    likelihood = 0
    for pred, ref in zip(predictions, targets):
        likelihood += log_prob_normal(pred, ref, likelihood_sd)
    result = likelihood / len(predictions)

    return result


###############################################################
# Name: Bayesian Linear Layer
# Function:
# Comment:
###############################################################
class BayesianLinearLayer(torch.nn.Module):
    def __init__(self, input_num, output_num, prior_std=1.):
        super(BayesianLinearLayer, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.prior_std = prior_std
        w_shape = (output_num, input_num)
        self.w_mean = Parameter(torch.randn(*w_shape))
        self.w_pho = Parameter(torch.randn(*w_shape).fill_(std_to_pho(prior_std)))

        b_shape = (output_num, 1)
        self.b_mean = Parameter(torch.randn(*b_shape).fill_(0))
        self.b_pho = Parameter(torch.randn(*b_shape).fill_(std_to_pho(prior_std)))
    
    # Forward Propagation
    def forward(self, x, is_prob=True):
        w_var = pho_to_std(self.w_pho)**2

        b_var = pho_to_std(self.b_pho)**2
        out = []
        for xi in x:
            xi = xi.unsqueeze(dim=1)
            gamma = torch.mm(self.w_mean, xi) + self.b_mean
            delta = torch.mm(w_var, xi**2) + b_var

            zeta = torch.randn(gamma.size(), device=x.device)      # device ???
            if is_prob:
                yi = gamma + delta.sqrt() * zeta      # (out_chans, 1)
            else:      # work as a normal linear layer with mean weights
                yi = gamma
            out.append(yi.squeeze(dim=1))

        return torch.stack(out, dim=0)      # (N, out_chans)

    def extra_repr(self):
        s = ('{input_num}, {output_num}')
        s += ', bias=True'
        if self.prior_std != 0.5:
            s += ', prior_std={prior_std}'
        
        return s.format(**self.__dict__)


###############################################################
# Name: Bayesian Neural Network
# Function:
# Comment:
###############################################################
class BayesianNeuralNetwork(torch.nn.Module):
    def __init__(self, input_units, hidden_units, output_units, prior_std=0.5, likelihood_sd=0.5):
        super(BayesianNeuralNetwork, self).__init__()
        self.input_layer = BayesianLinearLayer(input_units, hidden_units, prior_std)
        self.output_layer = BayesianLinearLayer(hidden_units, output_units, prior_std)
        self.likelihood_sd = likelihood_sd
        self.relu = nla_relu(True)
        self.prior_std = prior_std

    # Forward Propagation
    def forward(self, x, is_prob=True):
        res = self.input_layer.forward(x, is_prob)
        res = self.relu(res)
        res = self.output_layer.forward(res, is_prob)

        return res
    
    ##########################
    # kl_new_prior
    # Comment:predictions: KL divergence KL[params||prior] for a fully factorized Gaussian \
    # prior is given by prior_std at initialization of BNN
    ##########################
    def kl_new_prior(self):
        mean_temp = self.get_means()
        pho_temp = self.get_phos()
        assert len(mean_temp) == len(pho_temp)
        kl = 0
        for mean, pho in zip(mean_temp, pho_temp):
            kl += kl_div_p_q(mean, pho_to_std(pho), 0.0, self.prior_std)

        return kl

    def loss(self, predictions, targets, weight_kl=1.):
        kl = self.kl_new_prior()
        log_p_d_given_w = likelihood_criterion(predictions, targets, self.likelihood_sd)
        result = weight_kl * kl - log_p_d_given_w

        return result

    # ????????????????
    def kl_new_old(self):
        raise NotImplementedError

    def get_means(self):
        return [p[1] for p in self.named_parameters() if 'mean' in p[0]]

    def get_phos(self):
        return [p[1] for p in self.named_parameters() if 'pho' in p[0]]

    def loss_last_sample(self, inputs, targets):
        predictions = self.forward(inputs)
        loss = likelihood_criterion(predictions, targets, self.likelihood_sd)

        return loss

    ##########################
    # intrinsic reward
    # step_size 1e-2
    ##########################
    def kl_second_order_approx(self, step_size, inputs, targets):
        self.zero_grad()
        loss = self.loss_last_sample(inputs, targets)
        loss.backward()
        means = self.get_means()
        phos = self.get_phos()
        kl = 0
        # mu/pho have different shapes
        for mu, pho in zip(means, phos):
            grad_mu = mu.grad
            grad_pho = pho.grad
            pho = pho.detach()      # ????? no need to retain computation graph
            h_mu = 1 / ((1+pho.exp()).log()) ** 2
            h_pho = ((2*(2*pho).exp()) / (1+pho.exp())**2) * h_mu
            kl += torch.dot(grad_mu.pow(2).flatten(), 1 / h_mu.flatten())
            kl += torch.dot(grad_pho.pow(2).flatten(), 1 / h_pho.flatten())
            kl *= 0.5 * step_size**2

            return kl

    def get_parameters(self):
        mean_temp = self.get_means()
        pho_temp = self.get_phos()
        result = mean_temp + pho_temp

        return result

    def get_gradient_flatten(self, inputs, targets):
        pass

    def get_diag_hessian(self, param=None):
        pass

    def second_order_update(self, step_size):
        raise NotImplementedError
