# Author: Franciscon Valente Castro
# Email: francisco.valente@cimat.mx

import os
import sys
import math
import scipy
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import multiprocessing as mp
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

from toolkit.core.trajdataset import TrajDataset
from toolkit.core.trajlet import split_trajectories
from toolkit.test.load_all import get_datasets, all_dataset_names, get_trajlets



def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.5 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def Gauss_K(x, y, h):
    N = len(x)
    return math.exp(-np.linalg.norm(x - y) ** 2 / (2 * h ** 2)) / (2 * math.pi*h**2)**N

# Separates every trajectory in its observed and predicted trajlets
def obs_pred_trajectories(trajectories, separator=8, f_per_traj=20):
    N_t = len(trajectories)
    Trajm = []
    Trajp = []
    for tr in trajectories:
        Trajm.append(tr[range(separator), :])
        Trajp.append(tr[range(separator, f_per_traj), :])
    return Trajm, Trajp


# Weights for the GMM
def weights(Xm, k, h, Tobs=8, Tpred=12):
    N_t = len(Xm)
    aux = 0
    for idx in range(N_t):
        aux += Gauss_K(Xm[k], Xm[idx], h)
    w = []
    for idx in range(N_t):
        var = Gauss_K(Xm[k], Xm[idx], h) / aux
        w.append(var)

    return np.array(w)


def get_sample(Xp, w, M, h):
    probabilities = w / np.sum(w)
    l_sample = np.random.choice(range(len(Xp)), M, p=probabilities)
    sample = []
    for i in l_sample:
        sample.append(Xp[i])
    size = len(Xp[0]) * 2
    cov = h * np.identity(size)

    for i in range(len(sample)):
        s = sample[i].reshape(size)
        s = multivariate_normal.rvs(s, cov)
        s = s.reshape(int(size / 2), 2)
        sample[i] = s

    return sample


# Entropy estimation of the kth trajectory ones weights have been gotten
def entropy(Xp, k, h, w, M):
    # Sample size
    sample_size = 100

    # Get sample
    samples = get_sample(Xp, w, sample_size, h)

    # To numpy
    samples = np.array(samples)
    X = samples[:, -1, :].reshape([sample_size, 2])
    # X = samples.reshape([samples.shape[0],
    #                      samples.shape[1] * samples.shape[2]])

    # Calculate number of clusters
    num_clusters = find_number_of_clusters(X, plot=False,
                                           max_clusters=31)

    return num_clusters


def detect_separator(trajectories, secs):
    traj = trajectories[0]
    for i in range(len(traj)):
        if traj[i, 4] - traj[0, 4] > secs:
            break
    return i - 1


def get_entropies(opentraj_root, trajectories, dataset_name, M=30):
    # Load dataset
    N_t = len(trajectories)

    # Number of frames in observed and predicted trajlets
    Tobs = detect_separator(trajectories, 3.2)
    Tpred = len(trajectories[0]) - Tobs

    h = 0.5  # Bandwidth for Gaussian Kernel

    # Leave just the position information
    trajs = []
    for i in range(len(trajectories)):
        trajs.append(trajectories[i][:, 0:2])

    # Obtain observed and predicted trajlets
    Xm, Xp = obs_pred_trajectories(trajs, Tobs, Tpred + Tobs)

    # Estimate the entropy for every trajectory
    entropy_values = []
    for k in tqdm(range(N_t), dataset_name):
        w = weights(Xm, k, h)
        entropy_values.append(entropy(Xp, k, h, w, M))

    print('Done with the entropies of', dataset_name)
    return entropy_values


def entropies_set(opentraj_root, datasets_name, M=30):
    entropies_dir = os.path.join(opentraj_root, 'entropies')
    if not os.path.exists(entropies_dir):
        os.makedirs(entropies_dir)

    trajectories_set = get_trajlets(opentraj_root, datasets_name)

    entropy_values_set = {}
    for name in trajectories_set:
        trajlet_entropy_file = os.path.join(entropies_dir,
                                            name + str(M) + '-entropy.npy')
        if os.path.exists(trajlet_entropy_file):
            entropy_values = np.load(trajlet_entropy_file)
            print("loading entropies from: ", trajlet_entropy_file)
        else:
            entropy_values = get_entropies(opentraj_root,
                                           trajectories_set[name],
                                           name, M)
            np.save(trajlet_entropy_file, entropy_values)
            print("writing entropies ndarray into: ", trajlet_entropy_file)
        entropy_values_set.update({name: entropy_values})
    return entropy_values_set


def plot_eth(X, clusters, t):
    # Load reference image - ETH
    img_path = sys.argv[1] + '/datasets/UCY/students03/reference.png'
    img = plt.imread(img_path)

    # Homography matrix path
    H_path = sys.argv[1] + '/datasets/UCY/students03/H.txt'
    H = np.loadtxt(H_path)
    Hinv = np.linalg.inv(H)

    # Transform points to image space
    cat = np.vstack([X[:, 0], X[:, 1], np.ones_like(X[:, 0])]).T
    tCat = (Hinv @ cat.T).T

    # Get points in image
    x = tCat[:, 1] / tCat[:, 2]
    y = tCat[:, 0] / tCat[:, 2]

    # Fit GMM model to sample
    gmm = GaussianMixture(n_components=clusters).fit(X)
    labels = gmm.predict(X)
    plt.imshow(img)
    plt.scatter(x, y, c=labels,
                s=10, cmap='viridis')
    plt.title('Sample at t = {:.2f}'.format(t))
    plt.savefig(sys.argv[2] + '/clusters_t={:.2f}.png'.format(t))
    plt.close()


def get_sample_at_time(time, trajlets, interpolation=None, plot=True):
    # Sample
    sample_x = []
    sample_y = []
    trajectories = []
    new_interpolation = []

    # Get every trajectory
    for idx, traj in enumerate(trajlets):
        # Get x and y values
        x = traj[:, 0]
        y = traj[:, 1]

        # To list
        x = x.tolist()
        y = y.tolist()

        # Ajust cubic spline
        n = len(x)

        # Too small for cubic interpolation
        if n <= 3:
            continue

        # When no interpolation found
        if interpolation is None:

            # Interpolate and parametrize
            t = np.linspace(0, 1, n)
            x_curve = interp1d(t, x, kind='cubic')
            y_curve = interp1d(t, y, kind='cubic')

            # Add to new interpolations
            new_interpolation.append((x_curve, y_curve))

        else:
            # Use provided interpolations
            x_curve, y_curve = interpolation[idx]

        # Get sample
        sample_x.append(x_curve(time))
        sample_y.append(y_curve(time))
        trajectories.append(traj)

    # Plot reference
    if plot:
        print('Sample Size', len(sample_x))
        plt.scatter(sample_x, sample_y, color='red')
        plt.show()

    # Update interpolation when missing
    if interpolation is None:
        interpolation = new_interpolation

    return (np.array([sample_x, sample_y], dtype=np.float32).T,
            np.array(trajectories),
            interpolation)


def find_number_of_clusters(X, plot=True, max_clusters=21):
    # Find number of clusters
    n_components = np.arange(1, min(X.shape[0], max_clusters))
    models = [GaussianMixture(n, covariance_type='full',
                              random_state=0).fit(X)
              for n in n_components]

    if plot:
        plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
        plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.show()

    # Getting number of clusters using Bayesian Information Control
    index = np.argmin([m.bic(X) for m in models])
    clusters = n_components[index]

    return clusters


def fit_GMM(X, max_clusters=21):
    # Number of clusters
    clusters = find_number_of_clusters(X, plot=False, max_clusters=10)
    print('Number of clusters with BIC : ', clusters)

    # Fit GMM model to sample
    gmm = GaussianMixture(n_components=clusters).fit(X)
    labels = gmm.predict(X)

    return clusters, labels


def analyze_trajlets(trajlets, ds_name, max_clusters=21,
                     bandwidth=0.2, plot=False):
    # Detect number of clusters for each time
    times = np.arange(0.0, 1.0, .02)
    num_clusters = []
    entropy = []
    interpolation = None

    for t in tqdm(times, ds_name):

        # When no interpolation
        if interpolation is None:
            X, _, interpolation =\
                get_sample_at_time(t, trajlets, plot=False)
        else:
            X, _, interpolation =\
                get_sample_at_time(t, trajlets,
                                   interpolation=interpolation,
                                   plot=False)

        # Find number of clusters
        clusters = find_number_of_clusters(X,
                                           max_clusters=max_clusters,
                                           plot=False)
        num_clusters.append(clusters)

        # Plot GMM fit
        if plot and ds_name == 'UCY-Univ3':
            plot_eth(X, clusters, t)

        # Calculate entropy
        # estimate pdf using KDE with gaussian kernel
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
        log_p = kde.score_samples(X)  # returns log(p) of data sample
        p = np.exp(log_p)             # estimate p of data sample

        # Normalize p
        p = p / np.sum(p)

        # Calculate entropy
        ent = scipy.stats.entropy(p)    # evaluate entropy
        entropy.append(ent)

    # Plot estimated
    print('Times : ', times)
    print('Number of clusters :', num_clusters)
    print('Entropy :', entropy)

    return times, num_clusters, entropy


def analyze_dataset_loop(arguments, opentraj_root):
    ds_name, trajlets = arguments

    # Folder of numpy files
    print(opentraj_root)
    global_dir = os.path.join(opentraj_root, 'global')
    if not os.path.exists(global_dir):
        os.makedirs(global_dir)

    # Load or calculate indicators
    trajlet_global_file = os.path.join(global_dir, ds_name + '-global.npy')
    if os.path.exists(trajlet_global_file):
        global_values = np.load(trajlet_global_file)
        print("loading global indicators from: ", trajlet_global_file)
    else:
        # Get global number of clusters
        times, num_clusters, entropy =\
            analyze_trajlets(trajlets, ds_name, plot=False)
        global_values = np.array([times, num_clusters, entropy])

        # Save calculated values
        np.save(trajlet_global_file, global_values)
        print("writing global indicators ndarray into: ", trajlet_global_file)

    return (global_values[0, :], global_values[1, :], global_values[2, :])


def run(trajlets, opentraj_root, output_dir):
    dataset_names = list(trajlets.keys())

    # Map indices
    arguments = [(ds_name, trajlets[ds_name])
                 for ds_name in dataset_names]

    if args.execution == 'normal':
        # Analyze datasets (normal)
        results = [analyze_dataset_loop(arg, opentraj_root)
                   for arg in arguments]
    elif args.execution == 'parallelized':
        # Analyze datasets (parallelized)
        pool = mp.Pool(mp.cpu_count() - 2)
        results = pool.map(analyze_dataset_loop, arguments)
        pool.close()

    # Construct dataframe
    df = pd.DataFrame()
    for ds_name, res in zip(dataset_names, results):
        times, num_clusters, entropy = res
        df_i = pd.DataFrame({'title': ds_name,
                             'times': times,
                             'num_clusters': num_clusters,
                             'entropy': entropy})

        df = df.append(df_i)

    print("making motion plots ...")

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 5))

    # Number of clusters creation
    fig.add_subplot(211)
    sns.swarmplot(y='num_clusters', x='title', data=df, size=3)
    plt.xlabel('')
    plt.xticks([])

    # Entropy plot creation
    ax2 = fig.add_subplot(212)
    sns.swarmplot(y='entropy', x='title', data=df, size=3)
    plt.xlabel('')
    plt.xticks(rotation=-20)
    ax2.yaxis.label.set_size(9)
    ax2.xaxis.set_tick_params(labelsize=8)

    plt.subplots_adjust(wspace=0, hspace=.1)

    plt.savefig(os.path.join(output_dir, "filename.pdf"),
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    # Parser arguments
    parser = argparse.ArgumentParser(description='Calculate global '
                                                 'multimodality indicators.')
    parser.add_argument('--opentraj_root', '--root')
    parser.add_argument('--output_dir', '--output')
    parser.add_argument('--execution', '--exe',
                        default='normal',
                        choices=['normal', 'parallelized'],
                        help='pick a execution (default: "vae")')
    args = parser.parse_args()

    # Dataset names
    dataset_names = all_dataset_names

    # Get trajectories
    trajlets = get_trajlets(args.opentraj_root, dataset_names)
    run(trajlets, args.opentraj_root, args.output_dir)
