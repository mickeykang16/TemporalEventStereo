import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math

def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

def disp_warp(img, disp, padding_mode='border', interpolate_mode = 'bilinear'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    # assert disp.min() >= 0
    
    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode=interpolate_mode, padding_mode=padding_mode, align_corners=True)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode=interpolate_mode, padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask


def flow_warp(feature, flow, mask=False, padding_mode='zeros', interpolate_mode = 'bilinear'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2
    
    grid = coords_grid(b, h, w).to(flow.device) + flow # [B, 2, H, W]

    return bilinear_sampler(feature, grid, mode = interpolate_mode, mask=mask, padding_mode=padding_mode)


def coords_grid(batch, ht, wd, normalize=False):
    if normalize:  # [-1, 1]
        coords = torch.meshgrid(2 * torch.arange(ht) / (ht - 1) - 1,
                                2 * torch.arange(wd) / (wd - 1) - 1)
    else:
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1)  # [B, 2, H, W]

def bilinear_sampler(img, coords, mode='bilinear', mask=False, padding_mode='zeros'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    if coords.size(-1) != 2:  # [B, 2, H, W] -> [B, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)

    H, W = img.shape[-2:]
    # H = height if height is not None else img.shape[-2]
    # W = width if width is not None else img.shape[-1]

    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # To handle H or W equals to 1 by explicitly defining height and width
    if H == 1:
        assert ygrid.abs().max() < 1e-8
        H = 10
    if W == 1:
        assert xgrid.abs().max() < 1e-8
        W = 10

    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)
    # breakpoint()
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.squeeze(-1).float()

    return img