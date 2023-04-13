import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

from matrix_inversion import Inverse
from matrix_multiplication import Multiply
from matrix_utils import FlopsCounter, get_submatrices


# LU class


class LU(FlopsCounter):
    def __init__(self, l: int) -> None:
        super().__init__()
        self._multiply = Multiply(l)
        self._inverse = Inverse(l)

    def __call__(self, A: np.ndarray):
        self._multiply.reset_flops()
        self._inverse.reset_flops()
        L, U = self._lu(A)
        self._flops += self._multiply.flops + self._inverse.flops
        return L, U

    def _lu(self, A: np.ndarray):
        """
        Returns the results of the LU factorization.
        L (lower) matrix has ones on its diagonal.
        """
        assert A.shape[0] == A.shape[1]
        n = A.shape[0]

        if n == 1:
            return np.ones_like(A), A

        A11, A12, A21, A22 = get_submatrices(A)
        submatrices_flops = A11.shape[0] * A11.shape[1]

        L11, U11 = self._lu(A11)

        U11_inv = self._inverse(U11)
        L21 = self._multiply(A21, U11_inv)

        L11_inv = self._inverse(L11)
        U12 = self._multiply(L11_inv, A12)

        S = A22 - self._multiply(L21, U12)
        self._flops += submatrices_flops

        L22, U22 = self._lu(S)

        L12, U21 = np.zeros_like(A12), np.zeros_like(A21)

        L = np.vstack((np.hstack((L11, L12)), np.hstack((L21, L22))))
        U = np.vstack((np.hstack((U11, U12)), np.hstack((U21, U22))))

        return L, U


# Test functions


def run_tests(k_max: int, l: int, check=False):
    """
    Plots grid of graphs with measurement of time or floating-point operations of LU factorization
    for diffirent k and l parameters (2^k x 2^k is size of matrices, l is limit of recurrent multiplication)
    """
    lu = LU(l)

    measurements = {
        k: _lu_factorization_measurement(lu, k, tries=3, check=check)
        for k in range(2, k_max + 1)
    }

    _, axs = plt.subplots(1, 2, figsize=(20, 20))

    times = [tup[0] for tup in measurements.values()]
    flops = [tup[1] for tup in measurements.values()]

    axs[0].plot(list(range(2, k_max + 1)), times)
    axs[1].plot(list(range(2, k_max + 1)), flops)

    axs[0].set_title("Times [s] for different k (matrix shape is 2^k x 2^k)")
    axs[0].grid()

    axs[1].set_title("Flops for different k (matrix shape is 2^k x 2^k)")
    axs[1].grid()

    plt.show()


def _lu_factorization_measurement(lu: LU, k: int, tries=1, max_value=1.5, check=False):
    """
    For each try generate matrix with values from -max_value to + max_value
    and size 2^k x 2^k, then compute LU factorization.
    If check = True, compare result of L * U with A
    Returns mean time of LU factorization and number of floating-point operations
    """

    times, flopses = [], []
    for t in range(tries):
        A = (np.random.sample((2**k, 2**k)) * 2 - 1) * max_value

        # reset flops clock
        lu.reset_flops()
        # measure time
        begin = time.time()
        L, U = lu(A)
        finish = time.time()

        if check:
            det_L, det_U, det_A = (
                np.product(L.diagonal()),
                np.product(U.diagonal()),
                np.linalg.det(A),
            )
            print(
                f"Matrix size={2**k} test={t} det(L)={det_L} det(U)={det_U} det(input)={det_A}"
            )
            assert np.allclose(det_L, 1.0, atol=1e-6) and np.allclose(
                det_U, det_A, atol=1e-6
            ), "Check if determinents of results of custom LU factorization are correct"
            assert np.allclose(L @ U, A, atol=1e-6), "Check if L * U = A"

        times.append(finish - begin)
        flopses.append(lu.flops)

    return np.mean(times), np.mean(flopses)


# Main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=9)
    parser.add_argument("-l", type=int, default=8)
    parser.add_argument("--check", type=bool, default=False)

    args = parser.parse_args()
    k = args.k
    l = args.l
    check = args.check

    run_tests(k, l, check)
