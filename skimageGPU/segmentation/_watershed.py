import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi
from .._shared.utils import warn
from ..util import crop, regular_seeds, invert
from ._extrema_cy import _local_maxima


def _validate_connectivity(image_dim, connectivity, offset):
    """Convert any valid connectivity to a footprint and offset.
    Parameters
    ----------
    image_dim : int
        The number of dimensions of the input image.
    connectivity : int, array, or None
        The neighborhood connectivity. An integer is interpreted as in
        ``scipy.ndimage.generate_binary_structure``, as the maximum number
        of orthogonal steps to reach a neighbor. An array is directly
        interpreted as a footprint and its shape is validated against
        the input image shape. ``None`` is interpreted as a connectivity of 1.
    offset : tuple of int, or None
        The coordinates of the center of the footprint.
    Returns
    -------
    c_connectivity : array of bool
        The footprint (structuring element) corresponding to the input
        `connectivity`.
    offset : array of int
        The offset corresponding to the center of the footprint.
    Raises
    ------
    ValueError:
        If the image dimension and the connectivity or offset dimensions don't
        match.
    """
    if connectivity is None:
        connectivity = 1

    if cp.isscalar(connectivity):
        c_connectivity = ndi.generate_binary_structure(image_dim, connectivity)
    else:
        c_connectivity = cp.array(connectivity, bool)
        if c_connectivity.ndim != image_dim:
            raise ValueError("Connectivity dimension must be same as image")

    if offset is None:
        if any([x % 2 == 0 for x in c_connectivity.shape]):
            raise ValueError("Connectivity array must have an unambiguous "
                             "center")

        offset = cp.array(c_connectivity.shape) // 2

    return c_connectivity, offset


def _raveled_offsets_and_distances(
        image_shape,
        *,
        footprint=None,
        connectivity=1,
        center=None,
        spacing=None,
        order='C',
):
    """Compute offsets to neighboring pixels in raveled coordinate space.
    This function also returns the corresponding distances from the center
    pixel given a spacing (assumed to be 1 along each axis by default).
    Parameters
    ----------
    image_shape : tuple of int
        The shape of the image for which the offsets are being computed.
    footprint : array of bool
        The footprint of the neighborhood, expressed as an n-dimensional array
        of 1s and 0s. If provided, the connectivity argument is ignored.
    connectivity : {1, ..., ndim}
        The square connectivity of the neighborhood: the number of orthogonal
        steps allowed to consider a pixel a neighbor. See
        `scipy.ndimage.generate_binary_structure`. Ignored if footprint is
        provided.
    center : tuple of int
        Tuple of indices to the center of the footprint. If not provided, it
        is assumed to be the center of the footprint, either provided or
        generated by the connectivity argument.
    spacing : tuple of float
        The spacing between pixels/voxels along each axis.
    order : 'C' or 'F'
        The ordering of the array, either C or Fortran ordering.
    Returns
    -------
    raveled_offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their distance from the center.
    distances : ndarray
        The pixel distances corresponding to each offset.
    Notes
    -----
    This function will return values even if `image_shape` contains a dimension
    length that is smaller than `footprint`.
    Examples
    --------
    >>> off, d = _raveled_offsets_and_distances(
    ...         (4, 5), footprint=np.ones((4, 3)), center=(1, 1)
    ...         )
    >>> off
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    >>> d[0]
    1.0
    >>> d[-1]  # distance from (1, 1) to (3, 2)
    2.236...
    """
    ndim = len(image_shape)
    if footprint is None:
        footprint = ndi.generate_binary_structure(
            rank=ndim, connectivity=connectivity
        )
    if center is None:
        center = tuple(s // 2 for s in footprint.shape)

    if not footprint.ndim == ndim == len(center):
        raise ValueError(
            "number of dimensions in image shape, footprint and its"
            "center index does not match")

    offsets = cp.stack([(idx - c)
                        for idx, c in zip(cp.nonzero(footprint), center)],
                       axis=-1)

    if order == 'F':
        offsets = offsets[:, ::-1]
        image_shape = image_shape[::-1]
    elif order != 'C':
        raise ValueError("order must be 'C' or 'F'")

    # Scale offsets in each dimension and sum
    ravel_factors = image_shape[1:] + (1,)
    ravel_factors = cp.cumprod(ravel_factors[::-1])[::-1]
    raveled_offsets = (offsets * ravel_factors).sum(axis=1)

    # Sort by distance
    if spacing is None:
        spacing = cp.ones(ndim)
    weighted_offsets = offsets * spacing
    distances = cp.sqrt(cp.sum(weighted_offsets**2, axis=1))
    sorted_raveled_offsets = raveled_offsets[cp.argsort(distances)]
    sorted_distances = cp.sort(distances)

    # If any dimension in image_shape is smaller than footprint.shape
    # duplicates might occur, remove them
    if any(x < y for x, y in zip(image_shape, footprint.shape)):
        # np.unique reorders, which we don't want
        _, indices = cp.unique(sorted_raveled_offsets, return_index=True)
        sorted_raveled_offsets = sorted_raveled_offsets[cp.sort(indices)]
        sorted_distances = sorted_distances[cp.sort(indices)]

    # Remove "offset to center"
    sorted_raveled_offsets = sorted_raveled_offsets[1:]
    sorted_distances = sorted_distances[1:]

    return sorted_raveled_offsets, sorted_distances


def _offsets_to_raveled_neighbors(image_shape, footprint, center, order='C'):
    """Compute offsets to a samples neighbors if the image would be raveled.
    Parameters
    ----------
    image_shape : tuple
        The shape of the image for which the offsets are computed.
    footprint : ndarray
        The footprint (structuring element) determining the neighborhood
        expressed as an n-D array of 1's and 0's.
    center : tuple
        Tuple of indices to the center of `footprint`.
    order : {"C", "F"}, optional
        Whether the image described by `image_shape` is in row-major (C-style)
        or column-major (Fortran-style) order.
    Returns
    -------
    raveled_offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their distance from the center.
    Notes
    -----
    This function will return values even if `image_shape` contains a dimension
    length that is smaller than `footprint`.
    Examples
    --------
    >>> _offsets_to_raveled_neighbors((4, 5), np.ones((4, 3)), (1, 1))
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    >>> _offsets_to_raveled_neighbors((2, 3, 2), np.ones((3, 3, 3)), (1, 1, 1))
    array([ 2, -6,  1, -1,  6, -2,  3,  8, -3, -4,  7, -5, -7, -8,  5,  4, -9,
            9])
    """
    raveled_offsets = _raveled_offsets_and_distances(
        image_shape, footprint=footprint, center=center, order=order
    )[0]

    return raveled_offsets


def _resolve_neighborhood(footprint, connectivity, ndim,
                          enforce_adjacency=True):
    """Validate or create a footprint (structuring element).
    Depending on the values of `connectivity` and `footprint` this function
    either creates a new footprint (`footprint` is None) using `connectivity`
    or validates the given footprint (`footprint` is not None).
    Parameters
    ----------
    footprint : ndarray
        The footprint (structuring) element used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    ndim : int
        Number of dimensions `footprint` ought to have.
    enforce_adjacency : bool
        A boolean that determines whether footprint must only specify direct
        neighbors.
    Returns
    -------
    footprint : ndarray
        Validated or new footprint specifying the neighborhood.
    Examples
    --------
    >>> _resolve_neighborhood(None, 1, 2)
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]])
    >>> _resolve_neighborhood(None, None, 3).shape
    (3, 3, 3)
    """
    if footprint is None:
        if connectivity is None:
            connectivity = ndim
        footprint = ndi.generate_binary_structure(ndim, connectivity)
    else:
        # Validate custom structured element
        footprint = cp.asarray(footprint, dtype=bool)
        # Must specify neighbors for all dimensions
        if footprint.ndim != ndim:
            raise ValueError(
                "number of dimensions in image and footprint do not"
                "match"
            )
        # Must only specify direct neighbors
        if enforce_adjacency and any(s != 3 for s in footprint.shape):
            raise ValueError("dimension size in footprint is not 3")
        elif any((s % 2 != 1) for s in footprint.shape):
            raise ValueError("footprint size must be odd along all dimensions")

    return footprint


def _set_border_values(image, value, border_width=1):
    """Set edge values along all axes to a constant value.
    Parameters
    ----------
    image : ndarray
        The array to modify inplace.
    value : scalar
        The value to use. Should be compatible with `image`'s dtype.
    border_width : int or sequence of tuples
        A sequence with one 2-tuple per axis where the first and second values
        are the width of the border at the start and end of the axis,
        respectively. If an int is provided, a uniform border width along all
        axes is used.
    Examples
    --------
    >>> image = np.zeros((4, 5), dtype=int)
    >>> _set_border_values(image, 1)
    >>> image
    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])
    >>> image = np.zeros((8, 8), dtype=int)
    >>> _set_border_values(image, 1, border_width=((1, 1), (2, 3)))
    >>> image
    array([[1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]])
    """
    if cp.isscalar(border_width):
        border_width = ((border_width, border_width),) * image.ndim
    elif len(border_width) != image.ndim:
        raise ValueError('length of `border_width` must match image.ndim')
    for axis, npad in enumerate(border_width):
        if len(npad) != 2:
            raise ValueError('each sequence in `border_width` must have '
                             'length 2')
        w_start, w_end = npad
        if w_start == w_end == 0:
            continue
        elif w_start == w_end == 1:
            # Index first and last element in the current dimension
            sl = (slice(None),) * axis + ((0, -1),) + (...,)
            image[sl] = value
            continue
        if w_start > 0:
            # set first w_start entries along axis to value
            sl = (slice(None),) * axis + (slice(0, w_start),) + (...,)
            image[sl] = value
        if w_end > 0:
            # set last w_end entries along axis to value
            sl = (slice(None),) * axis + (slice(-w_end, None),) + (...,)
            image[sl] = value


def local_maxima(image, footprint=None, connectivity=None, indices=False,
                 allow_borders=True):
    """Find local maxima of n-dimensional array.
    The local maxima are defined as connected sets of pixels with equal gray
    level (plateaus) strictly greater than the gray levels of all pixels in the
    neighborhood.
    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    indices : bool, optional
        If True, the output will be a tuple of one-dimensional arrays
        representing the indices of local maxima in each dimension. If False,
        the output will be a boolean array with the same shape as `image`.
    allow_borders : bool, optional
        If true, plateaus that touch the image border are valid maxima.
    Returns
    -------
    maxima : ndarray or tuple[ndarray]
        If `indices` is false, a boolean array with the same shape as `image`
        is returned with ``True`` indicating the position of local maxima
        (``False`` otherwise). If `indices` is true, a tuple of one-dimensional
        arrays containing the coordinates (indices) of all found maxima.
    Warns
    -----
    UserWarning
        If `allow_borders` is false and any dimension of the given `image` is
        shorter than 3 samples, maxima can't exist and a warning is shown.
    See Also
    --------
    skimage.morphology.local_minima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima
    Notes
    -----
    This function operates on the following ideas:
    1. Make a first pass over the image's last dimension and flag candidates
       for local maxima by comparing pixels in only one direction.
       If the pixels aren't connected in the last dimension all pixels are
       flagged as candidates instead.
    For each candidate:
    2. Perform a flood-fill to find all connected pixels that have the same
       gray value and are part of the plateau.
    3. Consider the connected neighborhood of a plateau: if no bordering sample
       has a higher gray level, mark the plateau as a definite local maximum.
    Examples
    --------
    >>> from skimage.morphology import local_maxima
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])
    Find local maxima by comparing to all neighboring pixels (maximal
    connectivity):
    >>> local_maxima(image)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [ True, False, False, False, False, False,  True]])
    >>> local_maxima(image, indices=True)
    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))
    Find local maxima without comparing to diagonal pixels (connectivity 1):
    >>> local_maxima(image, connectivity=1)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [ True, False, False, False, False, False,  True]])
    and exclude maxima that border the image edge:
    >>> local_maxima(image, connectivity=1, allow_borders=False)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [False, False, False, False, False, False, False]])
    """
    image = cp.asarray(image, order="C")
    if image.size == 0:
        # Return early for empty input
        if indices:
            # Make sure that output is a tuple of 1 empty array per dimension
            return cp.nonzero(image)
        else:
            return cp.zeros(image.shape, dtype=bool)

    if allow_borders:
        # Ensure that local maxima are always at least one smaller sample away
        # from the image border
        image = cp.pad(image, 1, mode='constant', constant_values=image.min())

    # Array of flags used to store the state of each pixel during evaluation.
    # See _extrema_cy.pyx for their meaning
    flags = cp.zeros(image.shape, dtype=cp.uint8)

    _set_border_values(flags, value=3)

    if any(s < 3 for s in image.shape):
        # Warn and skip if any dimension is smaller than 3
        # -> no maxima can exist & footprint can't be applied
        warn(
            "maxima can't exist for an image with any dimension smaller 3 "
            "if borders aren't allowed",
            stacklevel=3
        )
    else:
        footprint = _resolve_neighborhood(footprint, connectivity,
                                          image.ndim)
        neighbor_offsets = _offsets_to_raveled_neighbors(
            image.shape, footprint, center=((1,) * image.ndim)
        )

        try:
            image_arr, flags_arr = np.array(image), np.array(flags)
            _local_maxima(image_arr.ravel(),
                          flags_arr.ravel(), neighbor_offsets)
            image, flags = cp.array(image_arr), cp.array(flags_arr)
        except TypeError:
            if image.dtype == cp.float16:
                # Provide the user with clearer error message
                raise TypeError("dtype of `image` is float16 which is not "
                                "supported, try upcasting to float32")
            else:
                raise  # Otherwise raise original message

    if allow_borders:
        # Revert padding performed at the beginning of the function
        flags = crop(flags, 1)
    else:
        # No padding was performed but set edge values back to 0
        _set_border_values(flags, value=0)

    if indices:
        return cp.nonzero(flags)
    else:
        return flags.view(bool)


def local_minima(image, footprint=None, connectivity=None, indices=False,
                 allow_borders=True):
    """Find local minima of n-dimensional array.
    The local minima are defined as connected sets of pixels with equal gray
    level (plateaus) strictly smaller than the gray levels of all pixels in the
    neighborhood.
    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    indices : bool, optional
        If True, the output will be a tuple of one-dimensional arrays
        representing the indices of local minima in each dimension. If False,
        the output will be a boolean array with the same shape as `image`.
    allow_borders : bool, optional
        If true, plateaus that touch the image border are valid minima.
    Returns
    -------
    minima : ndarray or tuple[ndarray]
        If `indices` is false, a boolean array with the same shape as `image`
        is returned with ``True`` indicating the position of local minima
        (``False`` otherwise). If `indices` is true, a tuple of one-dimensional
        arrays containing the coordinates (indices) of all found minima.
    See Also
    --------
    skimage.morphology.local_maxima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima
    Notes
    -----
    This function operates on the following ideas:
    1. Make a first pass over the image's last dimension and flag candidates
       for local minima by comparing pixels in only one direction.
       If the pixels aren't connected in the last dimension all pixels are
       flagged as candidates instead.
    For each candidate:
    2. Perform a flood-fill to find all connected pixels that have the same
       gray value and are part of the plateau.
    3. Consider the connected neighborhood of a plateau: if no bordering sample
       has a smaller gray level, mark the plateau as a definite local minimum.
    Examples
    --------
    >>> from skimage.morphology import local_minima
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = -1
    >>> image[3, 0] = -1
    >>> image[1:3, 4:6] = -2
    >>> image[3, 6] = -3
    >>> image
    array([[ 0,  0,  0,  0,  0,  0,  0],
           [ 0, -1, -1,  0, -2, -2,  0],
           [ 0, -1, -1,  0, -2, -2,  0],
           [-1,  0,  0,  0,  0,  0, -3]])
    Find local minima by comparing to all neighboring pixels (maximal
    connectivity):
    >>> local_minima(image)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [ True, False, False, False, False, False,  True]])
    >>> local_minima(image, indices=True)
    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))
    Find local minima without comparing to diagonal pixels (connectivity 1):
    >>> local_minima(image, connectivity=1)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [ True, False, False, False, False, False,  True]])
    and exclude minima that border the image edge:
    >>> local_minima(image, connectivity=1, allow_borders=False)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [False, False, False, False, False, False, False]])
    """
    return local_maxima(
        image=invert(image),
        footprint=footprint,
        connectivity=connectivity,
        indices=indices,
        allow_borders=allow_borders
    )


def _validate_inputs(image, markers, mask, connectivity):
    """Ensure that all inputs to watershed have matching shapes and types.
    Parameters
    ----------
    image : array
        The input image.
    markers : int or array of int
        The marker image.
    mask : array, or None
        A boolean mask, True where we want to compute the watershed.
    connectivity : int in {1, ..., image.ndim}
        The connectivity of the neighborhood of a pixel.
    Returns
    -------
    image, markers, mask : arrays
        The validated and formatted arrays. Image will have dtype float64,
        markers int32, and mask int8. If ``None`` was given for the mask,
        it is a volume of all 1s.
    Raises
    ------
    ValueError
        If the shapes of the given arrays don't match.
    """
    n_pixels = image.size
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = cp.ones(image.shape, bool)
    else:
        mask = cp.asanyarray(mask, dtype=bool)
        n_pixels = cp.sum(mask)
        if mask.shape != image.shape:
            message = (f'`mask` (shape {mask.shape}) must have same shape '
                       f'as `image` (shape {image.shape})')
            raise ValueError(message)
    if markers is None:
        markers_bool = local_minima(image, connectivity=connectivity) * mask
        footprint = ndi.generate_binary_structure(
            markers_bool.ndim, connectivity)
        markers = ndi.label(markers_bool, structure=footprint)[0]
    elif not isinstance(markers, (cp.ndarray, list, tuple)):
        # not array-like, assume int
        # given int, assume that number of markers *within mask*.
        markers = regular_seeds(image.shape,
                                int(markers / (n_pixels / image.size)))
        markers *= mask
    else:
        markers = cp.asanyarray(markers) * mask
        if markers.shape != image.shape:
            message = (f'`markers` (shape {markers.shape}) must have same '
                       f'shape as `image` (shape {image.shape})')
            raise ValueError(message)
    return (image.astype(cp.float64),
            markers.astype(cp.int32),
            mask.astype(cp.int8))


def watershed(image, markers=None, connectivity=1, offset=None, mask=None,
              compactness=0, watershed_line=False):
    """Find watershed basins in `image` flooded from given `markers`.
    Parameters
    ----------
    image : ndarray (2-D, 3-D, ...)
        Data array where the lowest value points are labeled first.
    markers : int, or ndarray of int, same shape as `image`, optional
        The desired number of markers, or an array marking the basins with the
        values to be assigned in the label matrix. Zero means not a marker. If
        ``None`` (no markers given), the local minima of the image are used as
        markers.
    connectivity : ndarray, optional
        An array with the same number of dimensions as `image` whose
        non-zero elements indicate neighbors for connection.
        Following the scipy convention, default is a one-connected array of
        the dimension of the image.
    offset : array_like of shape image.ndim, optional
        offset of the connectivity (one offset per dimension)
    mask : ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    compactness : float, optional
        Use compact watershed [3]_ with given compactness parameter.
        Higher values result in more regularly-shaped watershed basins.
    watershed_line : bool, optional
        If watershed_line is True, a one-pixel wide line separates the regions
        obtained by the watershed algorithm. The line has the label 0.
        Note that the method used for adding this line expects that
        marker regions are not adjacent; the watershed line may not catch
        borders between adjacent marker regions.
    Returns
    -------
    out : ndarray
        A labeled matrix of the same type and shape as markers
    See Also
    --------
    skimage.segmentation.random_walker : random walker segmentation
        A segmentation algorithm based on anisotropic diffusion, usually
        slower than the watershed but with good results on noisy data and
        boundaries with holes.
    Notes
    -----
    This function implements a watershed algorithm [1]_ [2]_ that apportions
    pixels into marked basins. The algorithm uses a priority queue to hold
    the pixels with the metric for the priority queue being pixel value, then
    the time of entry into the queue - this settles ties in favor of the
    closest marker.
    Some ideas taken from
    Soille, "Automated Basin Delineation from Digital Elevation Models Using
    Mathematical Morphology", Signal Processing 20 (1990) 171-182
    The most important insight in the paper is that entry time onto the queue
    solves two problems: a pixel should be assigned to the neighbor with the
    largest gradient or, if there is no gradient, pixels on a plateau should
    be split between markers on opposite sides.
    This implementation converts all arguments to specific, lowest common
    denominator types, then passes these to a C algorithm.
    Markers can be determined manually, or automatically using for example
    the local minima of the gradient of the image, or the local maxima of the
    distance function to the background for separating overlapping objects
    (see example).
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29
    .. [2] http://cmm.ensmp.fr/~beucher/wtshed.html
    .. [3] Peer Neubert & Peter Protzel (2014). Compact Watershed and
           Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation
           Algorithms. ICPR 2014, pp 996-1001. :DOI:`10.1109/ICPR.2014.181`
           https://www.tu-chemnitz.de/etit/proaut/publications/cws_pSLIC_ICPR.pdf
    Examples
    --------
    The watershed algorithm is useful to separate overlapping objects.
    We first generate an initial image with two overlapping circles:
    >>> x, y = np.indices((80, 80))
    >>> x1, y1, x2, y2 = 28, 28, 44, 52
    >>> r1, r2 = 16, 20
    >>> mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    >>> mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    >>> image = np.logical_or(mask_circle1, mask_circle2)
    Next, we want to separate the two circles. We generate markers at the
    maxima of the distance to the background:
    >>> from scipy import ndimage as ndi
    >>> distance = ndi.distance_transform_edt(image)
    >>> from skimage.feature import peak_local_max
    >>> max_coords = peak_local_max(distance, labels=image,
    ...                             footprint=np.ones((3, 3)))
    >>> local_maxima = np.zeros_like(image, dtype=bool)
    >>> local_maxima[tuple(max_coords.T)] = True
    >>> markers = ndi.label(local_maxima)[0]
    Finally, we run the watershed on the image and markers:
    >>> labels = watershed(-distance, markers, mask=image)
    The algorithm works also for 3-D images, and can be used for example to
    separate overlapping spheres.
    """
    image, markers, mask = _validate_inputs(image, markers, mask, connectivity)
    connectivity, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset)

    # pad the image, markers, and mask so that we can use the mask to
    # keep from running off the edges
    pad_width = [(p, p) for p in offset]
    image = cp.pad(image, pad_width, mode='constant')
    mask = cp.pad(mask, pad_width, mode='constant').ravel()
    output = cp.pad(markers, pad_width, mode='constant')

    flat_neighborhood = _offsets_to_raveled_neighbors(
        image.shape, connectivity, center=offset)
    marker_locations = cp.flatnonzero(output)
    image_strides = cp.array(image.strides, dtype=cp.intp) // image.itemsize

    _watershed_cy.watershed_raveled(image.ravel(),
                                    marker_locations, flat_neighborhood,
                                    mask, image_strides, compactness,
                                    output.ravel(),
                                    watershed_line)

    output = crop(output, pad_width, copy=True)

    return output
