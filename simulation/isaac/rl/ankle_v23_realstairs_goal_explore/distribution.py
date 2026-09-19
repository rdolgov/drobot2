"""Bounded Beta actions parameterized by a direct mean logit and concentration."""
import torch
from rsl_rl.modules.distribution import BetaDistribution

def parameters(output):
    mean_logit, log_extra = torch.unbind(output,dim=-2)
    proportion = torch.sigmoid(mean_logit)
    extra = torch.exp(log_extra.clamp(-4.,8.))
    return 1.+proportion*extra,1.+(1.-proportion)*extra

class MeanModule(torch.nn.Module):
    def __init__(self,scale,offset):
        super().__init__()
        self.scale,self.offset=scale,offset
    def forward(self,output):
        alpha,beta=parameters(output)
        return alpha/(alpha+beta)*self.scale+self.offset

class MeanStairBeta(BetaDistribution):
    def __init__(self,output_dim,action_range=(-1.,1.)):
        if output_dim!=16 or tuple(action_range)!=(-1.,1.):
            raise ValueError('This isolated stair arm uses 16 bounded actions')
        super().__init__(output_dim,action_range)
    def update(self,mlp_output):
        self._alpha,self._beta=parameters(mlp_output)
        self._distribution=torch.distributions.Beta(self._alpha,self._beta)
    def deterministic_output(self,mlp_output):
        alpha,beta=parameters(mlp_output)
        return alpha/(alpha+beta)*self._range_scale+self._range_offset
    def as_deterministic_output_module(self):
        return MeanModule(self._range_scale,self._range_offset)
