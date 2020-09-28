# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from pykalman import KalmanFilter
import numpy as np


class KalmanModel:
    def __init__(self, dt, n_dim=2, n_iter=4):
        self.n_iter = n_iter
        self.n_dim = n_dim

        # Const-acceleration Model
        self.A = np.array([[1, dt, dt ** 2],
                           [0, 1, dt],
                           [0, 0, 1]])

        self.C = np.array([[1, 0, 0]])

        self.Q = np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                           [dt ** 4 / 8., dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 3 / 6., dt ** 2 / 2, dt / 1]]) * 0.5

        # =========== Const-velocity Model ================
        # self.A = [[1, t],
        #           [0, 1]]
        #
        # self.C = [[1, 0]}
        #
        # q = 0.0005
        # self.Q = [[q, 0],
        #           [0, q/10]]
        # =================================================

        r = 1
        self.R = np.array([[r]])

        self.kf = [KalmanFilter(transition_matrices=self.A, observation_matrices=self.C,
                                transition_covariance=self.Q, observation_covariance=self.R) for _ in range(n_dim)]

    def filter(self, measurement):
        filtered_means = []
        for dim in range(self.n_dim):
            f = self.kf[dim].em(measurement[:, dim], n_iter=self.n_iter)
            (filtered_state_means, filtered_state_covariances) = f.filter(measurement[:, dim])
            filtered_means.append(filtered_state_means)
        filtered_means = np.stack(filtered_means)
        return filtered_means[:, :, 0].T, filtered_means[:, :, 1].T

    def smooth(self, measurement):
        smoothed_means = []
        if measurement.shape[0] == 1:
            return measurement, np.zeros((1, 2))
        for dim in range(self.n_dim):
            f = self.kf[dim].em(measurement[:, dim], n_iter=self.n_iter)
            (smoothed_state_means, smoothed_state_covariances) = f.smooth(measurement[:, dim])
            smoothed_means.append(smoothed_state_means)
        smoothed_means = np.stack(smoothed_means)
        return smoothed_means[:, :, 0].T, smoothed_means[:, :, 1].T


# TODO
def test_kalman():
    # index = 20
    # loc_measurement = pos_data[index]
    # vel_measurement = vel_data[index]
    #
    # dt = 1 / fps
    # kf = KalmanModel(dt=dt, n_iter=8)
    # filtered_pos, filtered_vel = kf.filter(loc_measurement)
    # smoothed_pos, smoothed_vel = kf.smooth(loc_measurement)
    #
    #
    # plt.subplot(1,2,1)
    # plt.plot(loc_measurement[0, 0], loc_measurement[0, 1], 'mo', markersize=5, label='Start Point')
    # plt.plot(loc_measurement[:, 0], loc_measurement[:, 1], 'r', label='Observation')
    # plt.plot(filtered_pos[:, 0], filtered_pos[:, 1], 'y--', label='Filter')
    # plt.plot(smoothed_pos[:, 0], smoothed_pos[:, 1], 'b--', label='Smoother')
    # plt.legend()
    #
    # plt.subplot(1,2,2)
    # plt.title("Velocity")
    # plt.plot(smoothed_vel[:, 0], 'b', label='Smoothed Vx')
    # plt.plot(smoothed_vel[:, 1], 'b', label='Smoothed Vy')
    #
    # plt.plot(vel_measurement[:, 0], 'g', label='Observed Vx')
    # plt.plot(vel_measurement[:, 1], 'g', label='Observed Vy')
    # plt.legend()
    #
    # plt.show()
    return


if __name__ == '__main__':
    test_kalman()
