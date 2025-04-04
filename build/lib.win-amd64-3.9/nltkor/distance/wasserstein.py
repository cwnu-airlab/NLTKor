import numpy as np

"""
source code = https://visualstudiomagazine.com/articles/2021/08/16/wasserstein-distance.aspx
By James McCaffrey
"""

from nltkor.make_requirement import make_requirement
try:
    import torch
except ImportError:
    requirement = ['torch']
    file_path = make_requirement(requirement)
    raise Exception(f"""
    Need to install Libraries, please pip install below libraries
    \t pip install torch
    Or, use pip install requirement.txt
    \t  pip install -r {file_path}
    """)


class WassersteinDistance:
    def __init__(self) -> None:
        pass

    def first_nonzero(self, vec):
        dim = len(vec)
        for i in range(dim):
            if vec[i] > 0.0:
                return i
        return -1  # no empty cells found

    def move_dirt(self, dirt, di, holes, hi):
    # move as much dirt at [di] as possible to h[hi]
        if dirt[di] <= holes[hi]:   # use all dirt
            flow = dirt[di]
            dirt[di] = 0.0            # all dirt got moved
            holes[hi] -= flow         # less to fill now
        elif dirt[di] > holes[hi]:  # use just part of dirt
            flow = holes[hi]          # fill remainder of hole
            dirt[di] -= flow          # less dirt left
            holes[hi] = 0.0           # hole is filled
        dist = np.abs(di - hi)
        return flow * dist          # work

    def compute_wasserstein(self, p, q):
        if "torch" in str(type(p)):
            p = p.numpy()
        if "torch" in str(type(q)):
            q = q.numpy()

        dirt = np.copy(p)
        holes = np.copy(q)
        tot_work = 0.0

        while True:  # TODO: add sanity counter check
            from_idx = self.first_nonzero(dirt)
            to_idx = self.first_nonzero(holes)
            if from_idx == -1 or to_idx == -1:
                break
            work = self.move_dirt(dirt, from_idx, holes, to_idx)
            tot_work += work
        return tot_work

    def kullback_leibler(self, p, q):
        n = len(p)
        sum = 0.0
        for i in range(n):
            sum += p[i] * np.log(p[i] / q[i])
        return sum

    def compute_kullback(self, p, q):
        if "torch" in str(type(p)):
            p = p.numpy()
        if "torch" in str(type(q)):
            q = q.numpy()
        a = self.kullback_leibler(p, q)
        b = self.kullback_leibler(q, p)
        return a + b

    def compute_jesson_shannon(self, p, q):
        if "torch" in str(type(p)):
            p = p.numpy()
        if "torch" in str(type(q)):
            q = q.numpy()

        a = self.kullback_leibler(p, (p + q)/2)
        b = self.kullback_leibler(q, (p + q)/2)
        return (a + b)/2


def demo():
    print("\nBegin Wasserstein distance demo ")

    P =  np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    Q1 = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
    Q2 = np.array([0.1, 0.1, 0.1, 0.1, 0.6])

    P = torch.from_numpy(P)
    Q1 = torch.from_numpy(Q1)
    Q2 = torch.from_numpy(Q2)
    kl_p_q1 = WassersteinDistance().compute_kullback(P, Q1)
    kl_p_q2 = WassersteinDistance().compute_kullback(P, Q2)

    wass_p_q1 = WassersteinDistance().compute_wasserstein(P, Q1)
    wass_p_q2 = WassersteinDistance().compute_wasserstein(P, Q2)

    jesson_p_q1 = WassersteinDistance().compute_jesson_shannon(P, Q1)
    jesson_p_q2 = WassersteinDistance().compute_jesson_shannon(P, Q2)

    print("\nKullback-Leibler distances: ")
    print("P to Q1 : %0.4f " % kl_p_q1)
    print("P to Q2 : %0.4f " % kl_p_q2)

    print("\nWasserstein distances: ")
    print("P to Q1 : %0.4f " % wass_p_q1)
    print("P to Q2 : %0.4f " % wass_p_q2)

    print("\nJesson-Shannon distances: ")
    print("P to Q1 : %0.4f " % jesson_p_q1)
    print("P to Q2 : %0.4f " % jesson_p_q2)

    print("\nEnd demo ")

if __name__ == "__main__":
    demo()
