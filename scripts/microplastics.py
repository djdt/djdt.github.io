import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from typing import List, Tuple


def parse_argv(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--cps", action="store_true", help="Convert data from CPS to counts."
    )
    parser.add_argument(
        "--trim",
        nargs=2,
        metavar="START END",
        type=int,
        help="Trim START, END points from the input file.",
    )
    parser.add_argument("--output", help="Output detecions to a file.")
    return parser.parse_args(argv[1:])


def cps_to_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convert counts per second into counts.

    Args:
        x: times in s
        y: counts per seond

    Returns:
        y in counts
    """
    dwell = np.around(np.mean(np.diff(x)), decimals=6)  # nearest microsecond
    return y * dwell


def poisson_limits(ub: float, epsilon: float = 0.5) -> Tuple[float, float]:
    """Calulate Yc and Yd for mean `ub`.

    If `ub` if lower than 5.0, the correction factor `epsilon` is added to `ub`.
    Lc and Ld can be calculated by adding `ub` to `Yc` and `Yd`.

    Args:
        ub: mean of background
        epsilon: low `ub` correct factor

    Returns:
        Yc, gross count critical value
        Yd, gross count detection limit

    References:
        Currie, L. A. (1968). Limits for qualitative detection and quantitative
            determination. Application to radiochemistry.
            Analytical Chemistry, 40(3), 586–593.
            doi:10.1021/ac60259a007
        Currie, L.A. On the detection of rare, and moderately rare, nuclear events.
            J Radioanal Nucl Chem 276, 285–297 (2008).
            https://doi.org/10.1007/s10967-008-0501-5
    """
    if ub < 5.0:  # 5 counts limit to maintain 0.05 alpha / beta (Currie 2008)
        ub += epsilon
    # Yc and Yd for paired distribution (Currie 1969)
    return 2.33 * np.sqrt(ub), 2.71 + 4.65 * np.sqrt(ub)


def accumulate_detections(
    y: np.ndarray,
    limit_detection: float,
    limit_accumulation: float,
    return_regions: bool = False,
) -> np.ndarray:
    """Returns an array of accumulated detections.

    Contiguous regions above `limit_accumulation` that contain at least one value above
    `limit_detection` are summed into the first point of the region.
    If `return_positions` then the start and end positions of valid regions
    is also returned.

    Args:
        y: array
        limit_detection: value for detection of region
        limit_accumulation: minimum accumulation value
        return_regions: also return starts and ends of regions

    Returns:
        array of summed detections
        if return_regions, start points of detected regions
        if return_regions, end points of detected regions
    """
    # Detections are points above Ld
    detections = np.nonzero(y > limit_detection)[0]
    # Find regions above Lc using diff of mask
    diff = np.concatenate(([0], np.diff((y > limit_accumulation).astype(np.int8))))
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]

    # Limit to regions where there is a detection
    idx = np.logical_and(starts <= detections[:, None], detections[:, None] < ends).any(
        axis=0
    )
    starts = starts[idx]
    ends = ends[idx]
    # Sum the regions
    sums = np.add.reduceat(y, np.stack((starts, ends), axis=-1).flat)[::2]

    if return_regions:
        return sums, starts, ends
    else:
        return sums


if __name__ == "__main__":
    args = parse_argv(sys.argv)
    x, y = np.genfromtxt(
        args.file, delimiter=",", skip_header=4, skip_footer=4, unpack=True
    )
    if args.trim is not None:
        y = y[args.trim[0] : -args.trim[1]]

    t0 = time.time()

    if args.cps:
        y = cps_to_counts(x, y)

    ub = np.mean(y)
    yc, yd = poisson_limits(ub)
    lc, ld = yc + ub, yd + ub

    acc, starts, ends = accumulate_detections(y, ld, lc, return_regions=True)
    # Remove μb from the accumulations
    acc -= ub * (ends - starts)

    # Select eveything under the Lc (non-detections)
    mask = y < lc
    nd_mean = np.mean(y[mask])  # Mean of the background without detections

    t1 = time.time()

    if args.output is not None:
        np.savetxt(args.output, acc)

    mean, median = np.mean(acc), np.median(acc)

    fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.hist(acc, bins=50, edgecolor="black")
    plt.title("Intensity distribution")
    plt.xlabel("Intensity")
    plt.ylabel("Detected Events")
    plt.axvline(x=median, c="red", ls="-.")
    plt.axvline(x=mean, c="green", ls="-.")
    plt.show()

    print("Detections:", acc.size)
    print("Mean intensity:", mean)
    print("Median intensity:", median)
    print("Background:", nd_mean)
    print("μb, Lc, Ld:", ub, lc, ld)
    print("Time taken:", t1 - t0, "s")
