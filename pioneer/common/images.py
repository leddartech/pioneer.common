from pioneer.common import interpolator as Ir

from matplotlib import cm, colors
import numpy as np

def to_image(v,h, scalars, indices, dtype = np.uint8):
    a = np.zeros((v, h), dtype, order='C')
    a[:,:].flat[indices] = scalars
    return a

def extrema_image(v, h, data, sort_field = 'amplitudes', sort_direction = -1
                  , other_fields = [], dtype = np.uint8, extrema = None):

    a = None

    if data.size > 0:
        #sort along negative amplitude, to obtain decreasing order
        rec = np.copy(data)
        rec[sort_field] *= sort_direction
        sorted_rec = np.sort(rec, order = ('indices', sort_field) )
        _, index = np.unique(sorted_rec['indices'], return_index = True) #assume unique will take the first encountered
        unique_rec = sorted_rec[index] #we got unique indices, we can get unique_rec, this will also give us min negative amplitudes

        if np.dtype(dtype) == np.uint8:
            if sort_direction < 0:
                div_ = min(extrema if extrema is not None else np.min(unique_rec[sort_field])
                           , -1e-3) #dynamic range
            else:
                div_ = max(extrema if extrema is not None else np.max(unique_rec[sort_field])
                           , 1e-3) #dynamic range
            div_ /= 255
        else:
            div_ = 1.0 if sort_direction > 0 else -1.0


        a = to_image(v,h, unique_rec[sort_field]/div_,unique_rec['indices'], dtype)


    if other_fields:
        other_results = {}
        for name in other_fields:
            if data.size > 0:
                other_results[name] = to_image(v,h,unique_rec[name],unique_rec['indices'], dtype)

        return a, other_results
    return a

def to_color_image(bw_image, colormap = 'viridis', vmin=0, vmax=None, log_scale=False):

    if vmax is None:
        if bw_image.dtype.kind == 'f':
            vmax = np.finfo(bw_image.dtype).max
        elif bw_image.dtype.kind == 'i':
            vmax = np.iinfo(bw_image.dtype).max
        else:
            raise RuntimeError("Unexpected type")
            
    if log_scale:
        norm = colors.LogNorm(1, vmax)
    else:
        norm = colors.Normalize(vmin, vmax)

    return (getattr(cm, colormap)(norm(bw_image))*255).astype(np.uint8)


def to_amp_dst_img(package):
    
    z = np.zeros((package['v'], package['h']), 'f4')

    if package['data'].size > 0:
        img, others = extrema_image(package['v'], package['h'], package['data']
                                                        , sort_field = 'amplitudes'
                                                        , sort_direction = -1
                                                        , other_fields = ['distances']
                                                        , dtype='f4')
        return np.flipud(img), np.flipud(others['distances'])
    else:
        return z, z


def accumulation_image(v, h, indices, weights, dtype = np.uint8, mode = 'normalized'):
    '''
        Can be used to produce a "sum of scalars" image
        Arguments:
            v,h,indices: n_rows, n_cols, channel indices
            weights: the scalar to sum (e.g. distances or amplitudes)
            mode: 'normalized', 'mean'
    '''
    acc = np.bincount(indices, weights, minlength = v*h)

    if mode == 'normalized':
        div_ = np.max(acc)

        if np.dtype(dtype) == np.uint8:
            div_ /= 255
        
        acc /= div_
        
    elif mode == 'mean':
        divs = np.bincount(indices, None, minlength = v*h)
        mask = divs > 0
        acc[mask] /= divs[mask]

    return to_image(v,h, acc, np.arange(v*h), dtype)

def mask_to_points(img, mask):
    '''
        Transforms a mask on an image to 3d points.
        Usage: mask_to_points(img, img > 5) -> np.array([[3, 4, 5.15], [10, 12, 5.17], [23, 15, 5.67], ..., , [145, 152, 7.89]])
    '''
    coords = np.argwhere(mask)
    pts = np.zeros((coords.shape[0], 3), dtype = 'f4')
    pts[:, :2] = coords
    pts[:, 2] = img[mask]
    return pts

def echoes_mask(echoes, arg_sort_fn):
    """Get a boolean mask where each element is True if the echo is the first
    echo along its ray. The order is determined with the `arg_sort_fn` parameter.
    Otherwise the value is false.

    Arguments:
        echoes {np.array} -- The echoes as returned by LeddarPy.
        arg_sort_fn {callable} -- A callable that performs argsort on the echoes

    Returns:
        [np.ndarray] -- The boolean mask
    """
    # generate an index list corresponding to each echo
    idx = np.arange(len(echoes))

    # get the indices of the sorted echoes
    sort_idx = arg_sort_fn(echoes)

    # get the indices (channels) of the echoes in sorted order
    sorted_indices = echoes['indices'][sort_idx]

    # reorder the linear indices of the unsorted echoes
    sorted_orig_idx = idx[sort_idx]

    # keep a single channel. Since the channels are sorted by distance or some
    # other criteria the closest echo along each ray will be kept.
    _, unique_index = np.unique(sorted_indices, return_index=True)

    # Use the indices of the unique items to get the original indices of the
    # echoes array.
    unique_linear_index = sorted_orig_idx[unique_index]

    # create a boolean mask and keep only the echoes corresponding to the first
    # echo along each ray.
    mask = np.zeros((len(echoes),), dtype=np.bool)
    mask[unique_linear_index] = True
    return mask

def echoes_visibility_mask(echoes):
    """Get a boolean mask where each element is True if the echo is the first
    echo along its ray. Otherwise the value is false.

    Arguments:
        echoes {np.array} -- The echoes as returned by LeddarPy. The

    Returns:
        [np.ndarray] -- The boolean mask
    """
    arg_sort_fn = lambda ech: np.argsort(ech['distances'])
    return echoes_mask(echoes, arg_sort_fn)

def maximum_amplitude_mask(echoes):
    """Get a boolean mask where each element is True if the echo is the one with
    maximal amplitude along its ray. Otherwise the value is false.

    Arguments:
        echoes {np.array} -- The echoes as returned by LeddarPy. The

    Returns:
        [np.ndarray] -- The boolean mask
    """
    arg_sort_fn = lambda ech: np.argsort(-ech['amplitudes'])
    return echoes_mask(echoes, arg_sort_fn)

def echoes_to_image(shape, indices, values, mask=None, min_value=None,
                    dtype=None, flip_ud=False, flip_lr=False, rot90=False):
    """Convert values to an image format using the echoes indices. Typically
    values will be amplitudes or distances.

    Arguments:
        shape {tuple} -- The shape of the sensor image
        indices {np.ndarray} -- The echoes channels
        values {np.ndarray} -- The values to store in image format
        (i.e. amplitudes, distances, etc)

    Keyword Arguments:
        mask {np.ndarray} -- A validity mask. Useful to keep only the closest echo for each ray. (default: {None})
        min_value {float,int} -- The minimum value to use to fill holes. (default: {None})
        dtype {np.dtype} -- The data type to return. If `None` the dtype of values is used. (default: {None})
        flip_lr {bool} -- Flip the image from left to right (default: {False})
        rot90 {bool} -- Rotate the image by 90 degrees (default: {False})

    Returns:
        np.ndarray -- The image representation of values
    """

    if min_value is None:
        min_value = values.min()

    if dtype is None:
        dtype = values.dtype

    if mask is None:
        mask = np.ones(indices.shape, dtype=np.bool)

    assert indices.shape == values.shape, \
        'Indices and values must have the same shape'

    assert mask.shape == values.shape, \
        'Mask must be a boolean mask with the same shape as values'

    image = np.empty(shape, dtype=dtype)
    image.fill(min_value)

    masked_indices = indices[mask]
    masked_values = values[mask]

    image.flat[masked_indices] = masked_values

    if flip_ud:
        image = np.flipud(image)
    if rot90:
        image = np.rot90(image)
    if flip_lr:
        image = np.fliplr(image)

    return image

def traces_to_amplitude_image(v, h, data):
    return np.max(data, axis = 1).reshape(v, h).astype('u2')

def interpolate_missings(image):
    """This function calls the interpolator 'inpaintn' for filling the missings
        elements in 'image'.

        Note that value zero in image are replaced by np.nan
    """
    image[image==0]=np.nan
    return Ir.inpaintn(image,m=100, x0=None, alpha=2)