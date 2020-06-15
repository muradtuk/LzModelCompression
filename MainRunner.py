"""
Author: Murad Tukan
"""

import numpy as np
import MVEEApprox


class MainRunner(object):
    def __init__(self, P, max_iter=int(1e2)):
        self.max_iter = max_iter
        self.P = P
        self.cost_func = lambda x: np.linalg.norm(np.dot(P, x), ord=1)
        self.mvee = MVEEApprox.MVEEApprox(P, self.cost_func, self.max_iter)

    def l1RankApprox(self):
        ellipsoid, center = self.mvee.compute_approximated_MVEE()
        G = np.linalg.pinv(ellipsoid)
        return G, ellipsoid, center

    def reduceRank(self, G, ranks):
        Gs = np.empty(G.shape + (ranks.shape[0], ))
        _, D, V = np.linalg.svd(G, full_matrices=True)
        for idx,_ in enumerate(ranks):
            D_temp = D[:ranks[idx]]
            V_temp = V[:, :ranks[idx]]
            Gs[:, :, idx] = np.dot(np.diag(D_temp), V_temp.T)

        return Gs

    @staticmethod
    def main():
        P = np.random.randn(1000, 50)  # Insert data here
        main_runner = MainRunner(P)
        G, _, _ = main_runner.l1RankApprox()
        ranks = np.array([10, 20, 30])  # make sure it is a numpy array
        Gs = main_runner.reduceRank(G, ranks)  # This is what you want, here you will get d x rank x 1 (in case u wanted
        # a single rank)



if __name__ == '__main__':
    MainRunner.main()
