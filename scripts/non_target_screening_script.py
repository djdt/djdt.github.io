"""Non-target screening script.

Non-target analysis and characterisation of nanoparticles in spirits via single particle
    ICP-ToF-MS
Raquel Gonzalez de Vega, Thomas E. Lockwood, Lhiam Paton, Lukas Schlatt, David Clases

The functions in this script are taken from SPCal version 1.1.2,
available at https://github.com/djdt/spcal.

Usage
-----

To screen data for potential NP signals first load the data as a numpy array
of shape (events, elements). Limits for each element can be calculated using the
``limit_for_element`` function::

    limits = [limit_for_element(data[:, i] for i in data.shape[1])]

These limits are then passed to the ``non_target_screen`` function with the desired
detection ppm (particles per events) along with the data::

    idx = non_target_screen(data, ppm=25.0, limits=limits)

The function returns the indicies of elements (along axis 1) with enough detections.
Elements can be screened indiviually with the ``screen_element`` function.
"""

from statistics import NormalDist
from typing import List, Tuple

import numpy as np
from scipy.stats import lognorm, poisson
from spcal.detection import accumulate_detections


def _contiguous_regions(x: np.ndarray, limit: float | np.ndarray) -> np.ndarray:
    """Returns start and end points of regions in x that are greater than limit.
    Indexs to the start point and point after region.

    Args:
        x: array
        limit: minimum value in regions

    Returns:
        regions [start, end]
    """

    # Get start and end positions of regions above accumulation limit
    diff = np.diff((x > limit).astype(np.int8), prepend=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    # Stack into pairs of start, end. If no final end position set it as end of array.
    if starts.size != ends.size:
        # -1 for reduceat
        ends = np.concatenate((ends, [diff.size]))  # type: ignore
    return np.stack((starts, ends), axis=1)


def _label_regions(regions: np.ndarray, size: int) -> np.ndarray:
    """Label regions 1 ... regions.size. Unlabled areas are 0.

    Args:
        regions: from `get_contiguous_regions`
        size: size of array

    Returns:
        labeled regions
    """
    labels = np.zeros(size, dtype=np.int16)
    if regions.size == 0:
        return labels

    ix = np.arange(1, regions.shape[0] + 1)

    starts, ends = regions[:, 0], regions[:, 1]
    ends = ends[ends < size]
    # Set start, end pairs to +i, -i.
    labels[starts] = ix
    labels[ends] = -ix[: ends.size]  # possible over end
    # Cumsum to label
    labels = np.cumsum(labels)
    return labels


def accumulate_detections(
    y: np.ndarray,
    limit_accumulation: float | np.ndarray,
    limit_detection: float | np.ndarray,
    integrate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns an array of accumulated detections.

    Contiguous regions above ``limit_accumulation`` that contain at least one value
    above ``limit_detection`` are summed or integrated (sum - ``limit_accumulation``).

    Args:
        y: array
        limit_accumulation: minimum accumulation value(s)
        limit_detection: minimum detection value(s)
        integrate: integrate, otherwise sum

    Returns:
        summed detection regions
        labels of regions
        regions [starts, ends]
    """
    if np.any(limit_accumulation > limit_detection):
        raise ValueError("accumulate_detections: limit_accumulation > limit_detection.")

    regions = _contiguous_regions(y, limit_accumulation)
    indicies = regions.ravel()
    if indicies.size > 0 and indicies[-1] == y.size:
        indicies = indicies[:-1]

    # Get maximum in each region
    detections = np.logical_or.reduceat(y > limit_detection, indicies)[::2]
    # Remove regions without a max value above detection limit
    regions = regions[detections]
    indicies = regions.ravel()
    if indicies.size > 0 and indicies[-1] == y.size:
        indicies = indicies[:-1]

    # Sum regions
    if integrate:
        sums = np.add.reduceat(y - limit_accumulation, indicies)[::2]
    else:
        sums = np.add.reduceat(y, indicies)[::2]

    # Create a label array of detections
    labels = _label_regions(regions, y.size)

    return sums, labels, regions


def screen_element(
    x: np.ndarray,
    minimum_count_ppm: float,
    detection_threshold: float,
    mode: str = "events",
) -> bool:
    """Screen element for signals.

    Returns true if ``x`` has ``minimum_count_ppm`` ppm points or particles greater than
    provided detection_threshold.

    This function is adapted from SPCal version 1.1.2.

    Args:
        x: data
        minimum_count_ppm: minimum number of points above limit
        detection_threshold: threshold above which to be considered a detection,
        mode: method of detection, 'events' or 'detections'

    Returns:
        True data has signal else False
    """
    if mode == "events":
        count = np.count_nonzero(x > detection_threshold)
    elif mode == "detections":
        count = accumulate_detections(
            x, np.minimum(x.mean(), detection_threshold), detection_threshold
        )[0].size
    else:
        raise ValueError("screening mode must be 'events' or 'detections'")

    return count * 1e6 / x.size > minimum_count_ppm


def non_target_screen(
    x: np.ndarray,
    minimum_count_ppm: float,
    detection_thresholds: List[float],
    mode: str = "events",
) -> np.ndarray:
    """Screen data for potential NP signals.

    Finds signals with ``minimum_count_ppm`` ppm points or particles greater than
    provided limits.

    This function is adapted from SPCal version 1.1.2.

    Args:
        x: data of shape (events, elements)
        minimum_count_ppm: minimum number of points above limit
        detection_thresholds: thresholds for each element
        mode: method of detection, 'events' or 'detections'

    Returns:
        indices of elements with potential signals
    """

    idx = [
        screen_element(
            x[:, i],
            minimum_count_ppm=minimum_count_ppm,
            detection_threshold=detection_thresholds[i],
            mode=mode,
        )
        for i in range(x.shape[1])
    ]
    return np.flatnonzero(idx)


def sum_iid_lognormals(
    n: int | np.ndarray, mu: float, sigma: float, method: str = "FW"
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum of ``n`` identical independant log-normal distributions.

    The sum is approximated by another log-normal distribution, defined by
    the returned parameters. Uses the Fenton-Wilkinson approximation for good
    right-tail accuracy.

    Args:
        n: int or array of ints
        mu: log mean of the underlying distributions
        sigma: log stddev of the underlying distributions

    Returns:
        mu, sigma of the log-normal approximation

    References:
        L. F. Fenton, "The sum of lognormal probability distributions in scatter
            transmission systems," IRE Trans. Commun. Syst., vol. CS-8, pp. 57-67, 1960.
            https://doi.org/10.1109/TCOM.1960.1097606
    """
    # Fenton-Wilkinson
    sigma2_x = np.log(
        (np.exp(sigma**2) - 1.0) * (n * np.exp(2.0 * mu)) / (n * np.exp(mu)) ** 2
        + 1.0
    )
    mu_x = np.log(n * np.exp(mu)) + 0.5 * (sigma**2 - sigma2_x)
    return mu_x, np.sqrt(sigma2_x)


def compound_poisson_lognormal_quantile(
    q: float, lam: float, mu: float, sigma: float
) -> float:
    """Appoximation of a compound Poisson-Log-normal quantile.

    Calcultes the zero-truncated quantile of the distribution by appoximating the
    log-normal sum for each value ``k`` given by the Poisson distribution. The
    CDF is calculated for each log-normal, weighted by the Poisson PDF for ``k``.
    The quantile is taken from the sum of the CDFs.

    <1% error for lam < 50.0

    Args:
        q: quantile
        lam: mean of the Poisson distribution
        mu: log mean of the log-normal distribution
        sigma: log stddev of the log-normal distribution

    Returns:
        the ``q``th value of the compound Poisson-Log-normal
    """

    # A reasonable overestimate of the upper value
    uk = (
        int((lam + 1.0) * NormalDist().inv_cdf(1.0 - (1.0 - q) / 1e3) * np.sqrt(lam))
        + 1
    )
    k = np.arange(0, uk + 1)
    pdf = poisson.pmf(k, lam)
    cdf = np.cumsum(pdf)

    # Calculate the zero-truncated quantile
    q0 = (q - pdf[0]) / (1.0 - pdf[0])
    if q0 <= 0.0:  # The quantile is in the zero portion
        return 0.0
    # Trim values with a low probability
    valid = pdf > 1e-6
    weights = pdf[valid][1:]
    k = k[valid][1:]
    cdf = cdf[valid]
    # Re-normalize weights
    weights /= weights.sum()

    # Get the sum LN for each value of the Poisson
    mus, sigmas = sum_iid_lognormals(k, np.log(1.0) - 0.5 * sigma**2, sigma)

    # The quantile of the last log-normal, must be lower than this
    upper_q = lognorm.ppf(q, mus[-1], sigmas[-1])
    lower_q = k[np.argmax(cdf > q) - 1]

    xs = np.linspace(lower_q, upper_q, 1000)
    cdf = np.sum(
        [w * lognorm.cdf(xs, m, s) for w, m, s in zip(weights, mus, sigmas)],
        axis=0,
    )
    q = xs[np.argmax(cdf > q0)]
    return q


def limit_for_element(
    x: np.ndarray, gaussian_alpha: float = 1e-7, compound_poisson_alpha: float = 1e-6
) -> float:
    """Calculate the best limit (Gaussian or Compound Poisson) for element.

    Signals with no points below 5 cts are considered Gaussian.

    Adapated from SPCal version 1.1.2.

    Args:
        x: data for element
        gaussian_alpha: alpha value is Gaussian used
        compound_poisson_alpha: alpha value is Compound Poisson used

    Returns:
        detection threshold for element
    """
    mod = np.mod(x[(x > 0.0) & (x <= 5.0)], 1.0)
    if mod.size == 0:  # no points below 5 counts, Gaussian
        z = NormalDist().inv_cdf(1.0 - gaussian_alpha)
        return x.mean() + x.std() * z
    else:  # Compound Poisson
        sigma = 0.47  # fit for Nu Instruments Vitesse
        return compound_poisson_lognormal_quantile(
            (1.0 - compound_poisson_alpha),
            x.mean(),
            np.log(1.0) - 0.5 * sigma**2,
            sigma,
        )
