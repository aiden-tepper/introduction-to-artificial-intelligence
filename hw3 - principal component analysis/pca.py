from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)


def get_covariance(dataset):
    x = dataset
    x_t = np.transpose(x)
    dot = np.dot(x_t, x)
    cov = dot / (len(x)-1)
    return cov


def get_eig(S, m):
    n = len(S)
    vals_raw, vects_raw = eigh(S, subset_by_index=[n-m, n-1])
    vals = [[0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        vals[i][i] = vals_raw[m-1-i]
    reverse = list(range(m-1, -1, -1))
    vects = vects_raw[:, reverse]
    return vals, vects


def get_eig_perc(S, perc):
    vals_raw, vects_raw = eigh(S)
    total = sum(vals_raw)
    vals_non_diagonal = []
    vector_indices = []
    for i in range(len(vals_raw)):
        current_perc = vals_raw[i] / total
        if current_perc > perc:
            vals_non_diagonal.append(vals_raw[i])
            vector_indices.append(i)
    length = len(vals_non_diagonal)
    vals = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(length):
        vals[i][i] = vals_non_diagonal[length - 1 - i]
    vector_indices.reverse()
    vects = vects_raw[:, vector_indices]
    return vals, vects


def project_image(img, U):
    length = len(img)
    u_t = np.transpose(U)
    alpha = np.dot(u_t, img)
    proj = []
    for j in range(length):
        proj.append(np.dot(alpha, U[j]))
    return proj


def display_image(orig, proj):
    size = int(len(orig)**0.5)
    orig_resized = []
    proj_resized = []
    for i in range(size):
        orig_resized.append(orig[size*i:size*(i+1)])
        proj_resized.append(proj[size*i:size*(i+1)])
    orig_resized = np.transpose(orig_resized)
    proj_resized = np.transpose(proj_resized)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    img1 = ax1.imshow(orig_resized, aspect='equal')
    fig.colorbar(img1, ax=ax1)
    img2 = ax2.imshow(proj_resized, aspect='equal')
    fig.colorbar(img2, ax=ax2)
    plt.show()
