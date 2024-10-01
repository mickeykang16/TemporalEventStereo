import numpy as np

import torch
import pdb
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
    'outdoor_night': 19.651191,
    'outdoor_day': 19.635287
}


def disparity_to_depth(disparity_image):


    unknown_disparity = disparity_image == float('inf')
    depth_image = \
        FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (
        disparity_image + 1e-7)
    depth_image[unknown_disparity] = float('inf')
    return depth_image


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


def calc_error_test(est_disp=None, gt_disp=None, lb=None, ub=None):
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
    # mean_depth = torch.Tensor([0.])
    # median_depth = torch.Tensor([0.])
    epe = torch.Tensor([0.])

    if (not torch.is_tensor(est_disp)) or (not torch.is_tensor(gt_disp)):
        return {
            '1px': error1 * 100,
            'epe': epe
        }

    assert torch.is_tensor(est_disp) and torch.is_tensor(gt_disp)
    assert est_disp.shape == gt_disp.shape

    est_disp = est_disp.clone().cpu()
    gt_disp = gt_disp.clone().cpu()

    gt_mask = gt_disp > ub
    nan_gt_mask = torch.isnan(gt_disp)
    gt_disp[gt_mask] = float('inf')
    gt_disp[nan_gt_mask] = float('inf')

    estimated_depth = disparity_to_depth(est_disp)
    ground_truth_depth = disparity_to_depth(gt_disp)

    
    mean_depth = compute_absolute_error(estimated_depth, ground_truth_depth)[1]
    median_depth = compute_absolute_error(estimated_depth, ground_truth_depth, use_mean=False)[1]
    
    # est_mask = est_disp > ub
    # est_disp[est_mask] = float('inf')
    


    binary_error_map, one_pixel_error = compute_n_pixels_error(est_disp, gt_disp, n=1.0)
            
    
    mean_disparity_error = compute_absolute_error(est_disp, gt_disp)[1]

            
    
    mask = torch.ones(gt_disp.shape, dtype=torch.bool, device=gt_disp.device)
    if lb is not None:
        mask = mask & (gt_disp >= lb)
    if ub is not None:
        mask = mask & (gt_disp <= ub)
        # mask = mask & (est_disp <= ub)
        
    mask.detach_()
    if abs(mask.float().sum()) < 1.0:
        return {
            '1px': one_pixel_error,
            'mean_depth': mean_depth * 100,
            'median_depth': median_depth * 100,
            'epe': mean_disparity_error
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

    epe = abs_error.float().mean()

    # .mean() will get a tensor with size: torch.Size([]), after decorate with torch.Tensor, the size will be: torch.Size([1])
    return {
        '1px': one_pixel_error,
        'mean_depth': mean_depth * 100,
        'median_depth': median_depth * 100,
        'epe': mean_disparity_error,
    }


def compute_absolute_error(estimated_disparity,
                           ground_truth_disparity,
                           use_mean=True):
    """Returns pixel-wise and mean absolute error.

    Locations where ground truth is not avaliable do not contribute to mean
    absolute error. In such locations pixel-wise error is shown as zero.
    If ground truth is not avaliable in all locations, function returns 0.

    Args:
        ground_truth_disparity: ground truth disparity where locations with
                                unknow disparity are set to inf's.
        estimated_disparity: estimated disparity.
        use_mean: if True than use mean to average pixelwise errors,
                  otherwise use median.
    """
    absolute_difference = (estimated_disparity - ground_truth_disparity).abs()
    locations_without_ground_truth = torch.isinf(ground_truth_disparity)
    pixelwise_absolute_error = absolute_difference.clone()
    pixelwise_absolute_error[locations_without_ground_truth] = 0
    absolute_differece_with_ground_truth = absolute_difference[
        ~locations_without_ground_truth]
    if absolute_differece_with_ground_truth.numel() == 0:
        average_absolute_error = 0.0
    else:
        if use_mean:
            average_absolute_error = absolute_differece_with_ground_truth.mean(
            ).item()
        else:
            average_absolute_error = absolute_differece_with_ground_truth.median(
            ).item()
    return pixelwise_absolute_error, average_absolute_error



def compute_n_pixels_error(estimated_disparity, ground_truth_disparity, n=3.0):
    """Return pixel-wise n-pixels error and % of pixels with n-pixels error.

    Locations where ground truth is not avaliable do not contribute to mean
    n-pixel error. In such locations pixel-wise error is shown as zero.

    Note that n-pixel error is equal to one if
    |estimated_disparity-ground_truth_disparity| > n and zero otherwise.

    If ground truth is not avaliable in all locations, function returns 0.

    Args:
        ground_truth_disparity: ground truth disparity where locations with
                                unknow disparity are set to inf's.
        estimated_disparity: estimated disparity.
        n: maximum absolute disparity difference, that does not trigger
           n-pixel error.
    """
    locations_without_ground_truth = torch.isinf(ground_truth_disparity)
    more_than_n_pixels_absolute_difference = (
        estimated_disparity - ground_truth_disparity).abs().gt(n).float()
    pixelwise_n_pixels_error = more_than_n_pixels_absolute_difference.clone()
    pixelwise_n_pixels_error[locations_without_ground_truth] = 0.0
    more_than_n_pixels_absolute_difference_with_ground_truth = \
        more_than_n_pixels_absolute_difference[~locations_without_ground_truth]
    if more_than_n_pixels_absolute_difference_with_ground_truth.numel() == 0:
        percentage_of_pixels_with_error = 0.0
    else:
        percentage_of_pixels_with_error = \
            more_than_n_pixels_absolute_difference_with_ground_truth.mean(
                ).item() * 100
    return pixelwise_n_pixels_error, percentage_of_pixels_with_error