import cupy as cp
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type
from ..util import img_as_float
from ..color import rgb2lab
import sys
from math import exp, ceil, sqrt

DBL_MAX = sys.float_info.max


def _quickshift_cython(image, kernel_size, max_dist, return_tree, random_seed):
    """Segments image using quickshift clustering in Color-(x,y) space.
    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.
    Parameters
    ----------
    image : (width, height, channels) ndarray
        Input image.
    kernel_size : float
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool
        Whether to return the full segmentation hierarchy tree and distances.
    random_seed : {None, int, `numpy.random.Generator`}, optional
        If `random_seed` is None the `numpy.random.Generator` singleton
        is used.
        If `random_seed` is an int, a new ``Generator`` instance is used,
        seeded with `random_seed`.
        If `random_seed` is already a ``Generator`` instance then that instance
        is used.
        Random seed used for breaking ties.
    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
    """

    random_state = cp.random.default_rng(random_seed)

    dtype = cp.float64
    # TODO join orphaned roots?
    # Some nodes might not have a point of higher density within the
    # search window. We could do a global search over these in the end.
    # Reference implementation doesn't do that, though, and it only has
    # an effect for very high max_dist.

    # window size for neighboring pixels to consider
    inv_kernel_size_sqr = -0.5 / (kernel_size * kernel_size)
    kernel_width = ceil(3 * kernel_size)

    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    densities = cp.zeros((height, width), dtype=dtype)

    current_density, closest, dist, t = None, None, None, None
    r, c, r_, c_, channel, r_min, r_max, c_min, c_max = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    current_pixel_ptr = None

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(scale=0.00001, size=(height, width)).astype(
        dtype, copy=False
    )

    # default parent to self
    parent = cp.arange(width * height, dtype=cp.intp).reshape(height, width)
    dist_parent = cp.zeros((height, width), dtype=dtype)

    # compute densities

    for r in range(height):
        r_min = max(r - kernel_width, 0)
        r_max = min(r + kernel_width + 1, height)
        for c in range(width):
            c_min = max(c - kernel_width, 0)
            c_max = min(c + kernel_width + 1, width)
            for r_ in range(r_min, r_max):
                for c_ in range(c_min, c_max):
                    dist = 0
                    for channel in range(channels):
                        t = image[r, c, channel] - image[r_, c_, channel]
                        dist += t * t
                    t = r - r_
                    dist += t * t
                    t = c - c_
                    dist += t * t
                    densities[r, c] += exp(dist * inv_kernel_size_sqr)

    # find nearest node with higher density
    for r in range(height):
        r_min = max(r - kernel_width, 0)
        r_max = min(r + kernel_width + 1, height)
        for c in range(width):
            current_density = densities[r, c]
            closest = DBL_MAX
            c_min = max(c - kernel_width, 0)
            c_max = min(c + kernel_width + 1, width)
            for r_ in range(r_min, r_max):
                for c_ in range(c_min, c_max):
                    if densities[r_, c_] > current_density:
                        dist = 0
                        # We compute the distances twice since otherwise
                        # we get crazy memory overhead
                        # (width * height * windowsize**2)
                        for channel in range(channels):
                            t = image[r, c, channel] - image[r_, c_, channel]
                            dist += t * t
                        t = r - r_
                        dist += t * t
                        t = c - c_
                        dist += t * t
                        if dist < closest:
                            closest = dist
                            parent[r, c] = r_ * width + c_
            dist_parent[r, c] = sqrt(closest)

    dist_parent_flat = cp.array(dist_parent).ravel()
    parent_flat = cp.array(parent).ravel()

    # remove parents with distance > max_dist
    too_far = dist_parent_flat > max_dist
    parent_flat[too_far] = cp.arange(width * height)[too_far]
    old = cp.zeros_like(parent_flat)

    # flatten forest (mark each pixel with root of corresponding tree)
    while (old != parent_flat).any():
        old = parent_flat
        parent_flat = parent_flat[parent_flat]

    parent_flat = cp.unique(parent_flat, return_inverse=True)[1]
    parent_flat = parent_flat.reshape(height, width)

    if return_tree:
        return parent_flat, parent, dist_parent
    return parent_flat


def quickshift(
    image,
    ratio=1.0,
    kernel_size=5,
    max_dist=10,
    return_tree=False,
    sigma=0,
    convert2lab=True,
    random_seed=42,
    *,
    channel_axis=-1
):
    """Segments image using quickshift clustering in Color-(x,y) space.
    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.
    Parameters
    ----------
    image : (width, height, channels) ndarray
        Input image. The axis corresponding to color channels can be specified
        via the `channel_axis` argument.
    ratio : float, optional, between 0 and 1
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    kernel_size : float, optional
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float, optional
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool, optional
        Whether to return the full segmentation hierarchy tree and distances.
    sigma : float, optional
        Width for Gaussian smoothing as preprocessing. Zero means no smoothing.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. For this purpose, the input is assumed to be RGB.
    random_seed : int, optional
        Random seed used for breaking ties.
    channel_axis : int, optional
        The axis of `image` corresponding to color channels. Defaults to the
        last axis.
    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
    Notes
    -----
    The authors advocate to convert the image to Lab color space prior to
    segmentation, though this is not strictly necessary. For this to work, the
    image must be given in RGB format.
    References
    ----------
    .. [1] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008
    """

    image = img_as_float(cp.atleast_3d(image))
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if image.ndim > 3:
        raise ValueError("only 2D color images are supported")

    # move channels to last position as expected by the Cython code
    image = cp.moveaxis(image, source=channel_axis, destination=-1)

    if convert2lab:
        if image.shape[-1] != 3:
            ValueError("Only RGB images can be converted to Lab space.")
        image = rgb2lab(image)

    if kernel_size < 1:
        raise ValueError("`kernel_size` should be >= 1.")

    image = gaussian(image, [sigma, sigma, 0], mode="reflect", channel_axis=-1)
    image = cp.ascontiguousarray(image * ratio)

    segment_mask = _quickshift_cython(
        image,
        kernel_size=kernel_size,
        max_dist=max_dist,
        return_tree=return_tree,
        random_seed=random_seed,
    )
    return segment_mask
