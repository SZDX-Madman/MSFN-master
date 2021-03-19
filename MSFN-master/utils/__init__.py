#!/usr/bin/env python3
__all__ = ["load_module",'Laplace_filter' ]

import loguru
import copy
import sys
import os
import pkgutil
import keyword
import importlib
import importlib.machinery
import torch.nn.functional as F
import torch
import numpy as np
from scipy import ndimage

def load_module(fpath):
    fpath = os.path.realpath(fpath)
    mod_name = []
    for i in fpath.split(os.path.sep):
        v = str()
        for j in i:
            if not j.isidentifier() and not j.isdigit():
                j = '_'
            v += j
        if not v.isidentifier() or keyword.iskeyword(v):
            v = '_' + v
        mod_name.append(v)
    mod_name = '_'.join(mod_name)
    if mod_name in sys.modules:  # return if already loaded
        return sys.modules[mod_name]
    mod_dir = os.path.dirname(fpath)
    sys.path.append(mod_dir)
    old_mod_names = set(sys.modules.keys())
    try:
        final_mod = importlib.machinery.SourceFileLoader(
            mod_name, fpath).load_module()
    finally:
        sys.path.remove(mod_dir)
    sys.modules[mod_name] = final_mod
    return final_mod


def Laplace_filter(x):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    kernel = np.array([kernel, kernel, kernel], dtype=np.float32)
    kernel = np.array([kernel, kernel, kernel], dtype=np.float32)
    kernel = torch.from_numpy(kernel)
    if (torch.cuda.is_available()):
        kernel = kernel.cuda()

    return F.conv2d(x,kernel,None,1,1,)


