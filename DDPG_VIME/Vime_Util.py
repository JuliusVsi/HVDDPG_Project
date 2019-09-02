import numpy as np
import torch


def build_test_set(s_c, a_c, s_n, pool, is_norm=True):
    if is_norm:
        eps = 1e-8
        s_mean, s_std, a_mean, a_std = pool.get_mean_std()
        s_c = (s_c - s_mean) / (s_std + eps)
        a_c = (a_c - a_mean) / (a_std + eps)
        s_n = (s_n - s_mean) / (s_std + eps)
    temp_x = np.array(np.append(s_c, a_c))
    temp_y = np.array(s_n)
    res_x = torch.tensor(temp_x, dtype=torch.float32, device='cpu').unsqueeze(dim=0)
    res_y = torch.tensor(temp_y, dtype=torch.float32, device='cpu').unsqueeze(dim=0)

    return res_x, res_y


def update_kl_normalization(kls, pre_kl_medians):
    pre_kl_medians.append(np.median(kls))
    result = np.mean(pre_kl_medians)

    return result
