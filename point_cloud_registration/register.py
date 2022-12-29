from scipy import spatial
import itertools
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm


# canonical tetrahedrons
canonicals = np.array([
    [
        [ 0.677, -0.145, -0.04 ],
        [-0.613, -0.145, -0.04 ],
        [-0.097,  0.306, -0.04 ],
        [ 0.032, -0.016,  0.121]
    ],
    [
        [-0.471,  0.055,  0.152],
        [-0.429, -0.101,  0.245],
        [ 0.474,  0.015, -0.126],
        [ 0.427,  0.031, -0.271]
    ],
    [
        [-0.387,  0.027, -0.298],
        [-0.064, -0.345,  0.38 ],
        [ 0.064,  0.466,  0.178],
        [ 0.387, -0.149, -0.261]
    ],
    [
        [-0.117, -0.432,  0.237],
        [-0.223,  0.231,  0.35 ],
        [ 0.237, -0.247, -0.387],
        [ 0.102,  0.448, -0.2  ]
    ],
    [
        [-0.84 ,  0.044, -0.035],
        [ 0.095,  0.001,  0.047],
        [ 0.369, -0.038, -0.003],
        [ 0.377, -0.007, -0.009]
    ],
    [
        [-0.394, -0.33 ,  0.089],
        [ 0.1  , -0.085, -0.647],
        [ 0.109,  0.249,  0.27 ],
        [ 0.185,  0.166,  0.288]
    ]
])
canonicals -= canonicals.mean(axis=1, keepdims=True)
canonicals /= np.linalg.norm(canonicals, axis=(1,2), keepdims=True)


def make_tetras(points):
    """
    :param np.array points: n x 3 point cloud
    :return np.array tetras: m x 4 indices of compact tetrahedrons
    """
    RELATIVE_RADIUS = 1.3
    CUTOFF_QUANTILE = 0.1
    
    tree = spatial.KDTree(points)

    # compute search radius
    d4, _ = tree.query(points, [4])
    r = RELATIVE_RADIUS * np.quantile(d4, CUTOFF_QUANTILE)

    return np.array(list(set(
        itertools.chain.from_iterable(
            itertools.combinations(x, 4) 
            for x in tree.query_ball_point(points, r) 
            if len(x) >=4))))


def ortho_procrustes(A, B):
    U, D, V = np.linalg.svd(A.T @ B)
    return U @ V, D.sum()


def disparity(A, B):
    """
    :params A, B: - normalized tetrahedrons
    :return: Frobenius norm of (A - scale * B 
    """
    R, scale = ortho_procrustes(A, B)
    return np.square(A - scale * B @ R.T).sum()


def min_disparity(A, B):
    """
    same as disparity(A, B) except the tetra is assumed unordered
    """
    return min(
        disparity(np.array(t), B) 
        for t in itertools.permutations(A))


def make_normal_tetras(points):
    vertices = make_tetras(points)
    coords = points[vertices, :]
    means = coords.mean(axis=1, keepdims=True)
    p = coords - means
    norms = np.linalg.norm(p, axis=(1, 2), keepdims=True)
    return dict(
        coords=coords,
        means=means,
        norms=norms,
        norm_coords=p / norms)


def compute_canonical_features(tetras):
    """
    canonical features are the minimal divergences of each 
    tetrahedron from the canonical tetrahedrons
    """
    tetras['features'] = np.array([
        [min_disparity(p, c) for c in canonicals] 
        for p in tqdm(tetras['norm_coords'], desc="computing tetra features")])


def select_tetras(tetras, selection):
    tetras['features'] = tetras['features'][selection, :]
    tetras['coords'] = tetras['coords'][selection, :]
    tetras['means'] = tetras['means'][selection, :, :]
    tetras['norms'] = tetras['norms'][selection, :, :]
    tetras['norm_coords'] = tetras['norm_coords'][selection, :, :]


def remove_common_tetras(tetras, max_density_threshold=0.2):
    densities = gaussian_kde(tetras['features'].T)(tetras['features'].T)
    ix = densities < max_density_threshold * densities.max()
    select_tetras(tetras, ix)
    

def match_features(tetras1, tetras2):
    F1 = tetras1['features']
    F2 = tetras2['features']
    tree = spatial.KDTree(F1)
    d4, _ = tree.query(F2, [1])
    r = 1.1 * np.quantile(d4, 0.2)

    distances, matches = tree.query(F2, [1], distance_upper_bound=r)
    distances = distances[:,0]

    return distances, matches