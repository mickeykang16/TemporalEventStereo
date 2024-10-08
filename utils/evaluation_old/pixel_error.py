import numpy as np

import torch


def calc_error(est_disp=None, gt_disp=None, lb=None, ub=None):
    """
    Args:
        est_disp (Tensor): in [..., Height, Width] layout
        gt_disp (Tensor): in [..., Height, Width] layout
        lb (scalar): the lower bound of disparity you want to mask out
        ub (scalar): the upper bound of disparity you want to mask out
    Output:
        dict: the error of 1px, 2px, 3px, 5px, in percent,
            range [0,100] and average error epe
    """
    error1 = torch.Tensor([0.])
    error2 = torch.Tensor([0.])
    error3 = torch.Tensor([0.])
    error5 = torch.Tensor([0.])
    epe = torch.Tensor([0.])

    if (not torch.is_tensor(est_disp)) or (not torch.is_tensor(gt_disp)):
        return {
            '1px': error1 * 100,
            '2px': error2 * 100,
            '3px': error3 * 100,
            '5px': error5 * 100,
            'epe': epe
        }

    assert torch.is_tensor(est_disp) and torch.is_tensor(gt_disp)
    assert est_disp.shape == gt_disp.shape

    est_disp = est_disp.clone().cpu()
    gt_disp = gt_disp.clone().cpu()

    mask = torch.ones(gt_disp.shape, dtype=torch.bool, device=gt_disp.device)
    if lb is not None:
        mask = mask & (gt_disp >= lb)
    if ub is not None:
        mask = mask & (gt_disp <= ub)
        # mask = mask & (est_disp <= ub)
        
    mask.detach_()
    if abs(mask.float().sum()) < 1.0:
        return {
            '1px': error1 * 100,
            '2px': error2 * 100,
            '3px': error3 * 100,
            '5px': error5 * 100,
            'epe': epe
        }
    ## original error compute
    gt_disp = gt_disp[mask]
    est_disp = est_disp[mask]
    abs_error = torch.abs(gt_disp - est_disp)
    
    ## hh error compute for top k ###################
    # abs_error = torch.abs(gt_disp - est_disp) * mask
    # pix_sum = torch.gt(abs_error, 1)
    # # error_sum = abs_error.sum(1).sum(1)
    # error_sum = pix_sum.sum(1).sum(1)
    # mask_sum = mask.sum(1).sum(1)
    
    # per_img_error = error_sum / mask_sum
    
    # values, indexs = torch.topk(-per_img_error, 1343)
    # gt_disp = gt_disp[indexs, :, :]
    # est_disp = est_disp[indexs, :, :]
    # mask = mask[indexs, :, :]
    # abs_error = torch.abs(gt_disp - est_disp) * mask
    ###############################
    
    
    total_num = mask.float().sum()
    
    error1 = torch.sum(torch.gt(abs_error, 1).float()) / total_num
    error2 = torch.sum(torch.gt(abs_error, 2).float()) / total_num
    error3 = torch.sum(torch.gt(abs_error, 3).float()) / total_num
    error5 = torch.sum(torch.gt(abs_error, 5).float()) / total_num
    epe = abs_error.float().mean()

    # .mean() will get a tensor with size: torch.Size([]), after decorate with torch.Tensor, the size will be: torch.Size([1])
    return {
        '1px': torch.Tensor([error1 * 100]),
        '2px': torch.Tensor([error2 * 100]),
        '3px': torch.Tensor([error3 * 100]),
        '5px': torch.Tensor([error5 * 100]),
        'epe': torch.Tensor([epe]),
    }