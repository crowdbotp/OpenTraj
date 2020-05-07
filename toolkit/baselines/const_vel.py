import numpy as np


def predict_single(obsv, n_next, noise=[]):
    """
    :param obsv:    (T x D)
    :param n_next:
    :param noise:
    :return: pred  (n_next x D)
    """
    T, D = obsv.shape[0], obsv.shape[1]
    vel = (obsv[-1] - obsv[0]) / (T - 1)
    pred = [vel * i for i in range(1, n_next + 1)] + obsv[-1]
    pred = np.stack(pred)
    return pred


def predict_multiple(obsvs, n_next, sub_batches=[], noise=[]):
    """
    :param obsvs:  (N x T x D)
    :param n_next: number of steps to predict
    :param sub_batches: will be ignored, since interactions will not be taken into account
    :param noise:       will be ignored, since this is a deterministic function
    :return: (N x n_next x D)
    """
    N, T, D = obsvs.shape[0], obsvs.shape[1], obsvs.shape[2]
    vels = (obsvs[:, -1] - obsvs[:, 0]) / (T-1)
    preds = [vels*i for i in range(1, n_next+1)] + obsvs[:, -1]
    preds = np.stack(preds)
    return preds


# test
if __name__ == '__main__':
    obsv = np.array([[0,0], [1,1], [2,2], [3,3]]).astype(np.float)
    pred  = predict_single(obsv.reshape(-1, 2), 4)
    print(pred)
    preds = predict_multiple(obsv.reshape(1, -1, 2), 4)
    print(preds)


