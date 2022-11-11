import torch
import os
from collections import OrderedDict

from .carn import Net as CARN
from .carn_m import Net as CARNM
from ModelZoo import MODEL_DIR


def load_carn():
    carn_model = CARN(2)
    state_dict = torch.load(os.path.join(MODEL_DIR, 'carn.pth'), map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    carn_model.load_state_dict(new_state_dict)
    return carn_model

def load_carnm():
    carn_model = CARNM(2)
    state_dict = torch.load(os.path.join(MODEL_DIR, 'carn_m.pth'), map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    carn_model.load_state_dict(new_state_dict)
    return carn_model