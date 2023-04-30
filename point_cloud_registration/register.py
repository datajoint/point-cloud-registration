__version__ = 0.4

from scipy import spatial, stats
import itertools
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm


# canonical tetrahedrons
canonicals = np.array(
    [
        [
            [0.677, -0.145, -0.04],
            [-0.613, -0.145, -0.04],
            [-0.097, 0.306, -0.04],
            [0.032, -0.016, 0.121],
        ],
        [
            [-0.471, 0.055, 0.152],
            [-0.429, -0.101, 0.245],
            [0.474, 0.015, -0.126],
            [0.427, 0.031, -0.271],
        ],
        [
            [-0.387, 0.027, -0.298],
            [-0.064, -0.345, 0.38],
            [0.064, 0.466, 0.178],
            [0.387, -0.149, -0.261],
        ],
        [
            [-0.117, -0.432, 0.237],
            [-0.223, 0.231, 0.35],
            [0.237, -0.247, -0.387],
            [0.102, 0.448, -0.2],
        ],
        [
            [-0.84, 0.044, -0.035],
            [0.095, 0.001, 0.047],
            [0.369, -0.038, -0.003],
            [0.377, -0.007, -0.009],
        ],
        [
            [-0.394, -0.33, 0.089],
            [0.1, -0.085, -0.647],
            [0.109, 0.249, 0.27],
            [0.185, 0.166, 0.288],
        ],
    ]
)
canonicals -= canonicals.mean(axis=1, keepdims=True)
canonicals /= np.linalg.norm(canonicals, axis=(1, 2), keepdims=True)


def make_tetras(points, *, relative_radius=1.2, cutoff_quantile=0.2):
    """
    :param np.array points: n x 3 point coordinates
    :return: dict with fields: coords, means, norms, and norm_coords
    describing local tetrahedrons
    """
    # compute max tetrahedron diameter
    tree = spatial.KDTree(points)
    d4, _ = tree.query(points, [4])
    max_diameter = relative_radius * np.quantile(d4, cutoff_quantile)

    # Generate all tetrahedrons below max_diameter
    vertices = np.array(
        list(
            set(
                itertools.chain.from_iterable(
                    itertools.combinations(x, 4)
                    for x in tree.query_ball_point(points, max_diameter)
                    if len(x) >= 4
                )
            )
        )
    )
    coords = points[vertices, :]
    means = coords.mean(axis=1, keepdims=True)
    p = coords - means
    norms = np.linalg.norm(p, axis=(1, 2), keepdims=True)
    return dict(
        coords=coords,
        means=means,
        norms=norms,
        vertices=vertices,
        norm_coords=p / norms,
    )


def _remove_common_tetras(tetras, max_feature_density=0.2):
    """
    remove densest portions of the feature cloud where false matches are most common
    """
    densities = gaussian_kde(tetras["features"].T)(tetras["features"].T)
    ix = densities < max_feature_density * densities.max()
    select_tetras(tetras, ix)


def ortho_procrustes(A, B):
    U, D, V = np.linalg.svd(A.T @ B)
    return U @ V, D.sum()


def fit(A, B):
    """
    Minimize A @ transform + offset - B
    Least-squares best-fit transform that maps points A into points B
    :param A: Nxm numpy array of corresponding points
    :param B: Nxm numpy array of corresponding points
    :return: transform, offset to map  A @ transform + offset into B
    """
    ma = A.mean(axis=0)
    mb = B.mean(axis=0)
    na = np.linalg.norm(A - ma)
    nb = np.linalg.norm(B - mb)
    R, _ = ortho_procrustes((B - mb) / nb, (A - ma) / na)
    transform = nb / na * R.T
    offset = mb - nb / na * ma @ R.T
    return transform, offset


def _disparity(A, B):
    """
    :params A, B: 4x3 np.arrays -- normalized tetrahedrons
    :return: Frobenius norm error
    """
    R, scale = ortho_procrustes(A, B)
    return np.square(A - scale * B @ R.T).sum()


def _min_disparity(A, B):
    """
    :params A, B: normalized tetrahedrons as
    :return: disparity(A, B) minimized over all vertex orderings
    """
    return min(_disparity(np.array(t), B) for t in itertools.permutations(A))


def compute_features(tetras, progress_bar=False, exclude_common=False):
    """
    canonical features are the minimal divergences of each
    tetrahedron from the canonical tetrahedrons
    """
    progress = tqdm if progress_bar else lambda x: x
    tetras["features"] = np.array(
        [
            [_min_disparity(p, c) for c in canonicals]
            for p in progress(tetras["norm_coords"], desc="computing tetra features")
        ]
    )
    if exclude_common:
        _remove_common_tetras(tetras, max_feature_density=0.2)


def select_tetras(tetras, selection):
    tetras["vertices"] = tetras["vertices"][selection, :]
    tetras["features"] = tetras["features"][selection, :]
    tetras["coords"] = tetras["coords"][selection, :]
    tetras["means"] = tetras["means"][selection, :, :]
    tetras["norms"] = tetras["norms"][selection, :, :]
    tetras["norm_coords"] = tetras["norm_coords"][selection, :, :]


def _get_vertex_order(tetra, ref):
    """
    :param tetra, ref: nd.array 4x3 normalized tetrahedrons
    """
    assert tetra.shape == ref.shape == (4, 3)
    return np.array(
        min(
            itertools.permutations(range(4)),
            key=lambda i: _disparity(tetra[np.array(i)], ref),
        )
    )


def match_features(tetras1, tetras2, *, max_disparity=0.01):
    """
    order best matching features within tetras1 and tetras2
    """
    tree = spatial.KDTree(tetras1["features"])
    d, _ = tree.query(tetras2["features"], [1])
    distances, matches = tree.query(
        tetras2["features"], [1], distance_upper_bound=1.2 * np.quantile(d, 0.2)
    )
    distances = distances[:, 0]
    matches = matches[:, 0]

    # remove tetras2 with no matches in tetras1
    keep = ~np.isinf(distances)
    distances = distances[keep]
    matches = matches[keep]
    select_tetras(tetras2, keep)

    # sort vertices in tetras2
    for i in range(len(matches)):
        tetra, ref = tetras2["norm_coords"][i], tetras1["norm_coords"][matches[i]]
        order = _get_vertex_order(tetra, ref)
        tetras2["norm_coords"][i, :, :] = tetras2["norm_coords"][i, order, :]
        tetras2["coords"][i, :, :] = tetras2["coords"][i, order, :]
        tetras2["vertices"][i, :] = tetras2["vertices"][i, order]

    match_disparity = np.array(
        [
            _disparity(c1, c2)
            for c1, c2 in zip(tetras1["norm_coords"][matches], tetras2["norm_coords"])
        ]
    )

    # eliminate tetras2 with large disparities
    if max_disparity is not None:
        keep = match_disparity < max_disparity
        matches = matches[keep]
        select_tetras(tetras2, keep)
        match_disparity = match_disparity[keep]

    # eliminate duplicate matches in tetras2
    uniq, counts = np.unique(matches, return_counts=True)
    keep = np.ones_like(matches, dtype=bool)
    for duplicate_match in uniq[counts > 1]:
        ix = np.where(matches == duplicate_match)[0]
        np.delete(ix, np.argmin(match_disparity[ix]))
        keep[ix] = False
    matches = matches[keep]
    select_tetras(tetras2, keep)

    # reorder tetras1 to match
    select_tetras(tetras1, matches)


def register(points1, points2, *, max_disparity=0.01, progress_bar=False):
    """
    Find the affine transformation to map point cloud `points1` to `points2`.

    points1 @ transform + offset will put points1 into the space of points2

    :return: transform, offset, control_points
    """
    tetras1 = make_tetras(points1, relative_radius=1.3, cutoff_quantile=0.4)
    tetras2 = make_tetras(points2, relative_radius=1.3, cutoff_quantile=0.4)
    compute_features(tetras1, progress_bar=progress_bar)
    compute_features(tetras2, progress_bar=progress_bar)
    match_features(tetras1, tetras2, max_disparity=max_disparity)

    # fit matching points
    X1 = np.reshape(tetras1["coords"], newshape=(-1, 3))
    X2 = np.reshape(tetras2["coords"], newshape=(-1, 3))

    # fit matching points and eliminate outliers
    keep = np.ones(X1.shape[0], dtype=bool)
    for _ in range(5):
        transform, offset = fit(X1[keep], X2[keep])
        err = np.sum(np.square(X1[keep] @ transform + offset - X2[keep]), axis=1)
        if stats.skew(err) < 1.0:
            break
        keep[keep] = err < 1.2 * np.quantile(err, 0.8)

    control_points = np.array(
        list(
            set(
                (v1, v2)
                for v1, v2 in zip(
                    tetras1["vertices"].reshape(-1)[keep],
                    tetras2["vertices"].reshape(-1)[keep],
                )
            )
        )
    )

    return transform, offset, control_points
