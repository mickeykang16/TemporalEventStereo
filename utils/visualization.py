import numpy as np

def disp_err_to_color(disp_est, disp_gt):
    """
    Calculate the error map between disparity estimation and disparity ground-truth
    hot color -> big error, cold color -> small error
    Args:
        disp_est (numpy.array): estimated disparity map
            in (Height, Width) layout, range [0,inf]
        disp_gt (numpy.array): ground truth disparity map
            in (Height, Width) layout, range [0,inf]
    Returns:
        disp_err (numpy.array): disparity error map
            in (Height, Width, 3) layout, range [0,1]
    """
    """ matlab
    function D_err = disp_error_image (D_gt,D_est,tau,dilate_radius)
    if nargin==3
      dilate_radius = 1;
    end
    [E,D_val] = disp_error_map (D_gt,D_est);
    E = min(E/tau(1),(E./abs(D_gt))/tau(2));
    cols = error_colormap();
    D_err = zeros([size(D_gt) 3]);
    for i=1:size(cols,1)
      [v,u] = find(D_val > 0 & E >= cols(i,1) & E <= cols(i,2));
      D_err(sub2ind(size(D_err),v,u,1*ones(length(v),1))) = cols(i,3);
      D_err(sub2ind(size(D_err),v,u,2*ones(length(v),1))) = cols(i,4);
      D_err(sub2ind(size(D_err),v,u,3*ones(length(v),1))) = cols(i,5);
    end
    D_err = imdilate(D_err,strel('disk',dilate_radius));
    """
    # error color map with interval (0, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, inf)/3.0
    # different interval corresponds to different 3-channel projection
    cols = np.array(
        [
            [0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
            [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
            [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
            [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
            [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
            [3 / 3.0, 6 / 3.0, 254, 224, 144],
            [6 / 3.0, 12 / 3.0, 253, 174, 97],
            [12 / 3.0, 24 / 3.0, 244, 109, 67],
            [24 / 3.0, 48 / 3.0, 215, 48, 39],
            [48 / 3.0, float("inf"), 165, 0, 38]
        ]
    )

    # [0, 1] -> [0, 255.0]
    disp_est = disp_est.copy() *  255.0
    disp_gt = disp_gt.copy() * 255.0
    # get the error (<3px or <5%) map
    tau = [3.0, 0.05]
    E = np.abs(disp_est - disp_gt)

    not_empty = disp_gt > 0.0
    tmp = np.zeros_like(disp_gt)
    tmp[not_empty] = E[not_empty] / disp_gt[not_empty] / tau[1]
    E = np.minimum(E / tau[0], tmp)

    h, w = disp_gt.shape
    err_im = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    for col in cols:
        y_x = not_empty & (E >= col[0]) & (E <= col[1])
        err_im[y_x] = col[2:]

    # value range [0, 1], shape in [H, W 3]
    err_im = err_im.astype(np.float64) / 255.0

    return err_im