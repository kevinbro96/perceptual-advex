import torch
import torchvision.models as torchvision_models
from torchvision.models.utils import load_state_dict_from_url
import math
from torch import nn
from torch.nn import functional as F
from .vae import CVAE_s2
import pdb

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

def get_eps_params(base_eps, resol):
    eps_list = []
    max_list = []
    min_list = []
    for i in range(3):
        eps_list.append(torch.full((resol, resol), base_eps, device='cuda'))
        min_list.append(torch.full((resol, resol), 0., device='cuda'))
        max_list.append(torch.full((resol, resol), 255., device='cuda'))

    eps_t = torch.unsqueeze(torch.stack(eps_list), 0)
    max_t = torch.unsqueeze(torch.stack(max_list), 0)
    min_t = torch.unsqueeze(torch.stack(min_list), 0)
    return eps_t, max_t, min_t

def get_cifar_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), CIFAR_STD[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

class CIFARNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class CIFARINNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(self.mean)
        return x

class PixelModel(nn.Module):
    def __init__(self, model, resol):
        super().__init__()
        self.model = model
        self.transform = ImagenetTransform(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [0, 255]
        '''
        x = self.transform(x)
        # x is now normalized as the model expects
        x = self.model(x)
        return x

class CIAttack(nn.Module):
    def __init__(self, model,  vae_path, eps_max=1, step_size=None,  num_iterations=7,resol=32, CNN_embed_dim=64, norm='linf', rand_init=True, scale_each=False):
        super().__init__()
        self.resol = resol
        self.normalize = CIFARNORMALIZE(resol)
        self.innormalize = CIFARINNORMALIZE(resol)
        self.nb_its = num_iterations
        self.eps_max = eps_max
        if step_size is None:
            step_size = eps_max / (self.nb_its ** 0.5)
        self.step_size = step_size
        self.resol = resol
        self.norm = norm
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.model = model
        self.vae = CVAE_s2(d=8, z=CNN_embed_dim)
        self.vae = nn.DataParallel(self.vae)
        save_model = torch.load(vae_path)
        model_dict = self.vae.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        self.vae.load_state_dict(model_dict)
        self.vae.eval()

    def _init(self, shape, eps):
        if self.rand_init:
            if self.norm == 'linf':
                init = torch.rand(shape, dtype=torch.float32, device='cuda') * 2 - 1
            elif self.norm == 'l2':
                init = torch.randn(shape, dtype=torch.float32, device='cuda')
                init_norm = torch.norm(init.view(init.size()[0], -1), 2.0, dim=1)
                normalized_init = init / init_norm[:, None, None, None]
                dim = init.size()[1] * init.size()[2] * init.size()[3]
                rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/dim)
                init = normalized_init * rand_norms[:, None, None, None]
            else:
                raise NotImplementedError
            init = eps[:, None] * init
            init.requires_grad_()
            return init
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')

    def forward(self,img, labels):
        img = self.normalize(img)
        hi = self.vae(img, mode="x-hi")
        xi = self.vae(hi, mode="hi-xi")
        xd = img - xi
        base_eps = self.eps_max * torch.ones(hi.size()[0], device='cuda')
        step_size = self.step_size * torch.ones(hi.size()[0], device='cuda')
        hidden = hi.detach()
        hidden.requires_grad = True

        delta = self._init(hidden.size(), base_eps)
        adv_xi = self.vae(hidden + delta, mode="hi-xi")

        adv_sample = xd.detach() + adv_xi
        adv_sample = self.innormalize(adv_sample)
        s = self.model(adv_sample)
        if self.norm == 'l2':
            l2_max = base_eps
        for it in range(self.nb_its):
            loss = self.criterion(s, labels)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            grad = delta.grad.data

            if self.norm == 'linf':
                grad_sign = grad.sign()
                delta.data = delta.data + step_size[:, None] * grad_sign
                delta.data = torch.max(torch.min(delta.data, base_eps[:, None]), -base_eps[:, None])
                #delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
            elif self.norm == 'l2':
                batch_size = delta.data.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None]
                delta.data = delta.data + step_size[:, None] * normalized_grad
                l2_delta = torch.norm(delta.data.view(batch_size, -1), 2.0, dim=1)
                # Check for numerical instability
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), l2_max / l2_delta)
                delta.data *= proj_scale[:, None]
                #delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
            else:
                raise NotImplementedError

            if it != self.nb_its - 1:
                adv_xi = self.vae(hidden + delta, mode="hi-xi")
                adv_sample = xd.detach() + adv_xi
                adv_sample = self.innormalize(adv_sample)
                s = self.model(adv_sample)
                delta.grad.data.zero_()

        return adv_sample.detach()