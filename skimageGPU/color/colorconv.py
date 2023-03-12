import cupy as cp
from .._shared.utils import channel_as_last_axis, _supported_float_type
from ..util import dtype

xyz_from_rgb = cp.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)
illuminants = {
    "A": {
        "2": (1.098466069456375, 1, 0.3558228003436005),
        "10": (1.111420406956693, 1, 0.3519978321919493),
        "R": (1.098466069456375, 1, 0.3558228003436005),
    },
    "B": {
        "2": (0.9909274480248003, 1, 0.8531327322886154),
        "10": (0.9917777147717607, 1, 0.8434930535866175),
        "R": (0.9909274480248003, 1, 0.8531327322886154),
    },
    "C": {
        "2": (0.980705971659919, 1, 1.1822494939271255),
        "10": (0.9728569189782166, 1, 1.1614480488951577),
        "R": (0.980705971659919, 1, 1.1822494939271255),
    },
    "D50": {
        "2": (0.9642119944211994, 1, 0.8251882845188288),
        "10": (0.9672062750333777, 1, 0.8142801513128616),
        "R": (0.9639501491621826, 1, 0.8241280285499208),
    },
    "D55": {
        "2": (0.956797052643698, 1, 0.9214805860173273),
        "10": (0.9579665682254781, 1, 0.9092525159847462),
        "R": (0.9565317453467969, 1, 0.9202554587037198),
    },
    "D65": {
        "2": (0.95047, 1.0, 1.08883),  # This was: `lab_ref_white`
        "10": (0.94809667673716, 1, 1.0730513595166162),
        "R": (0.9532057125493769, 1, 1.0853843816469158),
    },
    "D75": {
        "2": (0.9497220898840717, 1, 1.226393520724154),
        "10": (0.9441713925645873, 1, 1.2064272211720228),
        "R": (0.9497220898840717, 1, 1.226393520724154),
    },
    "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0), "R": (1.0, 1.0, 1.0)},
}


def get_xyz_coords(illuminant, observer, dtype=float):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.
    Parameters
    ----------
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.
    dtype: dtype, optional
        Output data type.
    Returns
    -------
    out : array
        Array with 3 elements containing the XYZ coordinates of the given
        illuminant.
    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return cp.asarray(illuminants[illuminant][observer], dtype=dtype)
    except KeyError:
        raise ValueError(
            f"Unknown illuminant/observer combination "
            f"(`{illuminant}`, `{observer}`)"
        )


# Haematoxylin-Eosin-DAB colorspace
# From original Ruifrok's paper: A. C. Ruifrok and D. A. Johnston,
# "Quantification of histochemical staining by color deconvolution,"
# Analytical and quantitative cytology and histology / the International
# Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4,
# pp. 291-9, Aug. 2001.
rgb_from_hed = cp.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
hed_from_rgb = cp.linalg.inv(rgb_from_hed)

# Following matrices are adapted form the Java code written by G.Landini.
# The original code is available at:
# https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html

# Hematoxylin + DAB
rgb_from_hdx = cp.array([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776], [0.0, 0.0, 0.0]])
rgb_from_hdx[2, :] = cp.cross(rgb_from_hdx[0, :], rgb_from_hdx[1, :])
hdx_from_rgb = cp.linalg.inv(rgb_from_hdx)

# Feulgen + Light Green
rgb_from_fgx = cp.array(
    [
        [0.46420921, 0.83008335, 0.30827187],
        [0.94705542, 0.25373821, 0.19650764],
        [0.0, 0.0, 0.0],
    ]
)
rgb_from_fgx[2, :] = cp.cross(rgb_from_fgx[0, :], rgb_from_fgx[1, :])
fgx_from_rgb = cp.linalg.inv(rgb_from_fgx)

# Giemsa: Methyl Blue + Eosin
rgb_from_bex = cp.array(
    [
        [0.834750233, 0.513556283, 0.196330403],
        [0.092789, 0.954111, 0.283111],
        [0.0, 0.0, 0.0],
    ]
)
rgb_from_bex[2, :] = cp.cross(rgb_from_bex[0, :], rgb_from_bex[1, :])
bex_from_rgb = cp.linalg.inv(rgb_from_bex)

# FastRed + FastBlue +  DAB
rgb_from_rbd = cp.array(
    [
        [0.21393921, 0.85112669, 0.47794022],
        [0.74890292, 0.60624161, 0.26731082],
        [0.268, 0.570, 0.776],
    ]
)
rbd_from_rgb = cp.linalg.inv(rgb_from_rbd)

# Methyl Green + DAB
rgb_from_gdx = cp.array(
    [[0.98003, 0.144316, 0.133146], [0.268, 0.570, 0.776], [0.0, 0.0, 0.0]]
)
rgb_from_gdx[2, :] = cp.cross(rgb_from_gdx[0, :], rgb_from_gdx[1, :])
gdx_from_rgb = cp.linalg.inv(rgb_from_gdx)

# Hematoxylin + AEC
rgb_from_hax = cp.array(
    [[0.650, 0.704, 0.286], [0.2743, 0.6796, 0.6803], [0.0, 0.0, 0.0]]
)
rgb_from_hax[2, :] = cp.cross(rgb_from_hax[0, :], rgb_from_hax[1, :])
hax_from_rgb = cp.linalg.inv(rgb_from_hax)

# Blue matrix Anilline Blue + Red matrix Azocarmine + Orange matrix Orange-G
rgb_from_bro = cp.array(
    [
        [0.853033, 0.508733, 0.112656],
        [0.09289875, 0.8662008, 0.49098468],
        [0.10732849, 0.36765403, 0.9237484],
    ]
)
bro_from_rgb = cp.linalg.inv(rgb_from_bro)

# Methyl Blue + Ponceau Fuchsin
rgb_from_bpx = cp.array(
    [
        [0.7995107, 0.5913521, 0.10528667],
        [0.09997159, 0.73738605, 0.6680326],
        [0.0, 0.0, 0.0],
    ]
)
rgb_from_bpx[2, :] = cp.cross(rgb_from_bpx[0, :], rgb_from_bpx[1, :])
bpx_from_rgb = cp.linalg.inv(rgb_from_bpx)

# Alcian Blue + Hematoxylin
rgb_from_ahx = cp.array(
    [[0.874622, 0.457711, 0.158256], [0.552556, 0.7544, 0.353744], [0.0, 0.0, 0.0]]
)
rgb_from_ahx[2, :] = cp.cross(rgb_from_ahx[0, :], rgb_from_ahx[1, :])
ahx_from_rgb = cp.linalg.inv(rgb_from_ahx)

# Hematoxylin + PAS
rgb_from_hpx = cp.array(
    [[0.644211, 0.716556, 0.266844], [0.175411, 0.972178, 0.154589], [0.0, 0.0, 0.0]]
)
rgb_from_hpx[2, :] = cp.cross(rgb_from_hpx[0, :], rgb_from_hpx[1, :])
hpx_from_rgb = cp.linalg.inv(rgb_from_hpx)


def _prepare_colorarray(arr, force_copy=False, *, channel_axis=-1):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = cp.asanyarray(arr)

    if arr.shape[channel_axis] != 3:
        msg = (
            f"the input array must have size 3 along `channel_axis`, "
            f"got {arr.shape}"
        )
        raise ValueError(msg)

    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == cp.float32:
        _func = dtype.img_as_float32
    else:
        _func = dtype.img_as_float64
    return _func(arr, force_copy=force_copy)


@channel_as_last_axis()
def rgb2xyz(rgb, *, channel_axis=-1):
    """RGB to XYZ color space conversion.
    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.
        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ format. Same dimensions as input.
    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).
    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts from sRGB.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
    Examples
    --------
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _prepare_colorarray(rgb, channel_axis=-1).copy()
    mask = arr > 0.04045
    arr[mask] = cp.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return arr @ xyz_from_rgb.T.astype(arr.dtype)


@channel_as_last_axis()
def xyz2lab(xyz, illuminant="D65", observer="2", *, channel_axis=-1):
    """XYZ to CIE-LAB color space conversion.
    Parameters
    ----------
    xyz : (..., 3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.
        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in CIE-LAB format. Same dimensions as input.
    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., 3, ...).
    ValueError
        If either the illuminant or the observer angle is unsupported or
        unknown.
    Notes
    -----
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2lab
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_lab = xyz2lab(img_xyz)
    """
    arr = _prepare_colorarray(xyz, channel_axis=-1)

    xyz_ref_white = get_xyz_coords(illuminant, observer, arr.dtype)

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = cp.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16.0 / 116.0

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return cp.concatenate([x[..., cp.newaxis] for x in [L, a, b]], axis=-1)


@channel_as_last_axis()
def rgb2lab(rgb, illuminant="D65", observer="2", *, channel_axis=-1):
    """Conversion from the sRGB color space (IEC 61966-2-1:1999)
    to the CIE Lab colorspace under the given illuminant and observer.
    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.
        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in Lab format. Same dimensions as input.
    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).
    Notes
    -----
    RGB is a device-dependent color space so, if you use this function, be
    sure that the image you are analyzing has been mapped to the sRGB color
    space.
    This function uses rgb2xyz and xyz2lab.
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)
