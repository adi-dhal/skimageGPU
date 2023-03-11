import cupy as cp
from ..util.dtype import img_as_float
from .._shared.utils import _supported_float_type, check_nD
from cupyx.scipy import ndimage as ndi

SOBEL_SMOOTH = cp.array([1, 2, 1]) / 4


def _mask_filter_result(result, mask):
    """Return result after masking.
    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is not None:
        erosion_footprint = ndi.generate_binary_structure(mask.ndim, mask.ndim)
        mask = ndi.binary_erosion(mask, erosion_footprint, border_value=0)
        result *= mask
    return result


def _kernel_shape(ndim, dim):
    """Return list of `ndim` 1s except at position `dim`, where value is -1.
    Parameters
    ----------
    ndim : int
        The number of dimensions of the kernel shape.
    dim : int
        The axis of the kernel to expand to shape -1.
    Returns
    -------
    shape : list of int
        The requested shape.
    Examples
    --------
    >>> _kernel_shape(2, 0)
    [-1, 1]
    >>> _kernel_shape(3, 1)
    [1, -1, 1]
    >>> _kernel_shape(4, -1)
    [1, 1, 1, -1]
    """
    shape = [1, ] * ndim
    shape[dim] = -1
    return shape


def _reshape_nd(arr, ndim, dim):
    """Reshape a 1D array to have n dimensions, all singletons but one.
    Parameters
    ----------
    arr : array, shape (N,)
        Input array
    ndim : int
        Number of desired dimensions of reshaped array.
    dim : int
        Which dimension/axis will not be singleton-sized.
    Returns
    -------
    arr_reshaped : array, shape ([1, ...], N, [1,...])
        View of `arr` reshaped to the desired shape.
    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> arr = rng.random(7)
    >>> _reshape_nd(arr, 2, 0).shape
    (7, 1)
    >>> _reshape_nd(arr, 3, 1).shape
    (1, 7, 1)
    >>> _reshape_nd(arr, 4, -1).shape
    (1, 1, 1, 7)
    """
    kernel_shape = _kernel_shape(ndim, dim)
    return cp.reshape(arr, kernel_shape)


def _generic_edge_filter(image, *, smooth_weights, edge_weights=[1, 0, -1],
                         axis=None, mode='reflect', cval=0.0, mask=None):
    """Apply a generic, n-dimensional edge filter.
    The filter is computed by applying the edge weights along one dimension
    and the smoothing weights along all other dimensions. If no axis is given,
    or a tuple of axes is given the filter is computed along all axes in turn,
    and the magnitude is computed as the square root of the average square
    magnitude of all the axes.
    Parameters
    ----------
    image : array
        The input image.
    smooth_weights : array of float
        The smoothing weights for the filter. These are applied to dimensions
        orthogonal to the edge axis.
    edge_weights : 1D array of float, optional
        The weights to compute the edge along the chosen axes.
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::
            edge_mag = np.sqrt(sum([_generic_edge_filter(image, ..., axis=i)**2
                                    for i in range(image.ndim)]) / image.ndim)
        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.
    """
    ndim = image.ndim
    if axis is None:
        axes = list(range(ndim))
    elif cp.isscalar(axis):
        axes = [axis]
    else:
        axes = axis
    return_magnitude = (len(axes) > 1)

    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    output = cp.zeros(image.shape, dtype=image.dtype)

    for edge_dim in axes:
        kernel = _reshape_nd(edge_weights, ndim, edge_dim)
        smooth_axes = list(set(range(ndim)) - {edge_dim})
        for smooth_dim in smooth_axes:
            kernel = kernel * _reshape_nd(smooth_weights, ndim, smooth_dim)
        ax_output = ndi.convolve(image, kernel, mode=mode)
        if return_magnitude:
            ax_output *= ax_output
        output += ax_output

    if return_magnitude:
        output = cp.sqrt(output) / cp.sqrt(ndim)
    return output


def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """Find edges in an image using the Sobel filter.
    Parameters
    ----------
    image : array
        The input image.
    mask : array of bool, optional
        Clip the output image to this mask. (Values where mask=0 will be set
        to 0.)
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::
            sobel_mag = np.sqrt(sum([sobel(image, axis=i)**2
                                     for i in range(image.ndim)]) / image.ndim)
        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.
    Returns
    -------
    output : array of float
        The Sobel edge map.
    See also
    --------
    sobel_h, sobel_v : horizontal and vertical edge detection.
    scharr, prewitt, farid, skimage.feature.canny
    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.
    .. [2] https://en.wikipedia.org/wiki/Sobel_operator
    Examples
    --------
    >>> from skimage import data
    >>> from skimage import filters
    >>> camera = data.camera()
    >>> edges = filters.sobel(camera)
    """
    output = _generic_edge_filter(image, smooth_weights=SOBEL_SMOOTH,
                                  axis=axis, mode=mode, cval=cval)
    output = _mask_filter_result(output, mask)
    return output
