import cupy as cp
from .._shared.utils import warn
from ..util import img_as_float64
from .._shared.filters import gaussian
from .._shared import utils


def set_root(forest, n, root):
    """
    Set all nodes on a path to point to new_root.
    Given the example above, given n=9, root=6, it would "reconnect" the tree.
    so forest[9] = 6 and forest[8] = 6
    The ultimate goal is that all tree nodes point to the real root,
    which is element 1 in this case.
    """
    j = None
    while (forest[n] < n):
        j = forest[n]
        forest[n] = root
        n = j

    forest[n] = root
    return forest


def find_root(forest, n):
    """Find the root of node n.
    Given the example above, for any integer from 1 to 9, 1 is always returned
    """
    root = n
    while (forest[root] < root):
        root = forest[root]
    return root


def join_trees(forest, n,  m):
    """Join two trees containing nodes n and m.
    If we imagine that in the example tree, the root 1 is not known, we
    rather have two disjoint trees with roots 2 and 6.
    Joining them would mean that all elements of both trees become connected
    to the element 2, so forest[9] == 2, forest[6] == 2 etc.
    However, when the relationship between 1 and 2 can still be discovered later.
    """

    if (n != m):
        root = find_root(forest, n)
        root_m = find_root(forest, m)

        if (root > root_m):
            root = root_m

        forest = set_root(forest, n, root)
        forest = set_root(forest, m, root)
    return forest


def _felzenszwalb(image, scale=1, sigma=0.8,
                  min_size=20):
    """Felzenszwalb's efficient graph based segmentation for
    single or multiple channels.
    Produces an oversegmentation of a single or multi-channel image
    using a fast, minimum spanning tree based clustering on the image grid.
    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.
    Parameters
    ----------
    image : (N, M, C) ndarray
        Input image.
    scale : float, optional (default 1)
        Sets the observation level. Higher means larger clusters.
    sigma : float, optional (default 0.8)
        Width of Gaussian smoothing kernel used in preprocessing.
        Larger sigma gives smother segment boundaries.
    min_size : int, optional (default 20)
        Minimum component size. Enforced using postprocessing.
    Returns
    -------
    segment_mask : (N, M) ndarray
        Integer mask indicating segment labels.
    """

    if image.shape[2] > 3:
        warn(RuntimeWarning(
            "Got image with third dimension of %s. This image "
            "will be interpreted as a multichannel 2d image, "
            "which may not be intended." % str(image.shape[2])),
            stacklevel=3)

    image = img_as_float64(image)

    # rescale scale to behave like in reference implementation
    scale = float(scale) / 255.
    image = gaussian(image, sigma=[sigma, sigma, 0], mode='reflect',
                     channel_axis=-1)
    height, width = image.shape[:2]

    # compute edge weights in 8 connectivity:
    down_cost = cp.sqrt(cp.sum((image[1:, :, :] - image[:height-1, :, :]) *
                               (image[1:, :, :] - image[:height-1, :, :]), axis=-1))
    right_cost = cp.sqrt(cp.sum((image[:, 1:, :] - image[:, :width-1, :]) *
                                (image[:, 1:, :] - image[:, :width-1, :]), axis=-1))
    dright_cost = cp.sqrt(cp.sum((image[1:, 1:, :] - image[:height-1, :width-1, :]) *
                                 (image[1:, 1:, :] - image[:height-1, :width-1, :]), axis=-1))
    uright_cost = cp.sqrt(cp.sum((image[1:, :width-1, :] - image[:height-1, 1:, :]) *
                                 (image[1:, :width-1, :] - image[:height-1, 1:, :]), axis=-1))
    costs = cp.hstack([
        right_cost.ravel(), down_cost.ravel(), dright_cost.ravel(),
        uright_cost.ravel()]).astype(float)

    # compute edges between pixels:
    segments = cp.arange(width * height, dtype=cp.intp).reshape(height, width)
    down_edges = cp.column_stack(
        (segments[1:, :].ravel(), segments[:height-1, :].ravel()))
    right_edges = cp.column_stack(
        (segments[:, 1:].ravel(), segments[:, :width-1].ravel()))
    dright_edges = cp.column_stack((segments[1:, 1:].ravel(
    ), segments[:height-1, :width-1].ravel()))
    uright_edges = cp.column_stack(
        (segments[:height-1, 1:].ravel(), segments[1:, :width-1].ravel()))
    edges = cp.vstack([right_edges, down_edges, dright_edges, uright_edges])

    # initialize data structures for segment size
    # and inner cost, then start greedy iteration over edges.
    edge_queue = cp.argsort(costs)
    edges = cp.ascontiguousarray(edges[edge_queue])
    costs = cp.ascontiguousarray(costs[edge_queue])
    segments_p = segments.data
    edges_p = edges.data
    costs_p = costs.data
    segment_size = cp.ones(width * height, dtype=cp.intp)

    # inner cost of segments
    cint = cp.zeros(width * height)
    seg0, seg1, seg_new, e
    inner_cost0, inner_cost1
    num_costs = costs.size

    # set costs_p back one. we increase it before we use it
    # since we might continue before that.
    costs_p -= 1
    for e in range(num_costs):
        seg0 = find_root(segments_p, edges_p[0])
        seg1 = find_root(segments_p, edges_p[1])
        edges_p += 2
        costs_p += 1
        if seg0 == seg1:
            continue
        inner_cost0 = cint[seg0] + scale / segment_size[seg0]
        inner_cost1 = cint[seg1] + scale / segment_size[seg1]
        if costs_p[0] < min(inner_cost0, inner_cost1):
            # update size and cost
            segments_p = join_trees(segments_p, seg0, seg1)
            seg_new = find_root(segments_p, seg0)
            segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]
            cint[seg_new] = costs_p[0]

    # postprocessing to remove small segments
    edges_p = edges.data
    for e in range(num_costs):
        seg0 = find_root(segments_p, edges_p[0])
        seg1 = find_root(segments_p, edges_p[1])
        edges_p += 2
        if seg0 == seg1:
            continue
        if segment_size[seg0] < min_size or segment_size[seg1] < min_size:
            segments_p = join_trees(segments_p, seg0, seg1)
            seg_new = find_root(segments_p, seg0)
            segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]

    # unravel the union find tree
    flat = segments.ravel()
    old = cp.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    flat = cp.unique(flat, return_inverse=True)[1]
    return flat.reshape((height, width))


@utils.channel_as_last_axis(multichannel_output=False)
def felzenszwalb(image, scale=1, sigma=0.8, min_size=20, *,
                 channel_axis=-1):
    """Computes Felsenszwalb's efficient graph based image segmentation.
    Produces an oversegmentation of a multichannel (i.e. RGB) image
    using a fast, minimum spanning tree based clustering on the image grid.
    The parameter ``scale`` sets an observation level. Higher scale means
    less and larger segments. ``sigma`` is the diameter of a Gaussian kernel,
    used for smoothing the image prior to segmentation.
    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.
    For RGB images, the algorithm uses the euclidean distance between pixels in
    color space.
    Parameters
    ----------
    image : (width, height, 3) or (width, height) ndarray
        Input image.
    scale : float
        Free parameter. Higher means larger clusters.
    sigma : float
        Width (standard deviation) of Gaussian kernel used in preprocessing.
    min_size : int
        Minimum component size. Enforced using postprocessing.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
    References
    ----------
    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and
           Huttenlocher, D.P.  International Journal of Computer Vision, 2004
    Notes
    -----
        The `k` parameter used in the original paper renamed to `scale` here.
    Examples
    --------
    >>> from skimage.segmentation import felzenszwalb
    >>> from skimage.data import coffee
    >>> img = coffee()
    >>> segments = felzenszwalb(img, scale=3.0, sigma=0.95, min_size=5)
    """
    if channel_axis is None and image.ndim > 2:
        raise ValueError("This algorithm works only on single or "
                         "multi-channel 2d images. ")

    image = cp.atleast_3d(image)
    return _felzenszwalb(image, scale=scale, sigma=sigma,
                         min_size=min_size)
