import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

from matrix_multiplication import Multiply
from matrix_utils import FlopsCounter, get_submatrices


# Inverse class


class Inverse(FlopsCounter):
    def __init__(self, l: int) -> None:
        super().__init__()
        self._multiply = Multiply(l)

    def __call__(self, A: np.ndarray):
        return self._inverse(A)

    def _inverse(self, A: np.ndarray):
        """
        Returns the result of the matrix inversion
        """
        assert A.shape[0] == A.shape[1]
        n = A.shape[0]

        if n == 1:
            self._flops += 1
            return 1 / A

        self._multiply.reset_flops()

        A11, A12, A21, A22 = get_submatrices(A)
        submatrices_flops = A11.shape[0] * A11.shape[1]

        A11_inv = self._inverse(A11)

        S22 = A22 - self._multiply(self._multiply(A21, A11_inv), A12)
        self._flops += submatrices_flops

        S22_inv = self._inverse(S22)

        B11 = A11_inv + self._multiply(
            self._multiply(A11_inv, A12),
            self._multiply(self._multiply(S22_inv, A21), A11_inv),
        )
        self._flops += submatrices_flops

        B12 = self._multiply(self._multiply(-A11_inv, A12), S22_inv)
        self._flops += submatrices_flops

        B21 = self._multiply(self._multiply(-S22_inv, A21), A11_inv)
        self._flops += submatrices_flops

        B22 = S22_inv

        self._flops += self._multiply.flops

        B = np.vstack((np.hstack((B11, B12)), np.hstack((B21, B22))))

        return B


# Test functions


def run_tests(k_max: int, l: int, check=False):
    """
    Plots grid of graphs with measurement of time or floating-point operations of inversion
    for diffirent k and l parameters (2^k x 2^k is size of matrices, l is limit of recurrent multiplication)
    """
    inverse = Inverse(l)

    measurements = {
        k: _inversion_measurement(inverse, k, check=check) for k in range(2, k_max + 1)
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


def _inversion_measurement(
    inverse: Inverse, k: int, tries=1, max_value=10, check=False
):
    """
    For each try generate matrix with values from -max_value to + max_value
    and size 2^k x 2^k, then inverse it.
    If check = True, compare result of inversion with result of numpy inversion function
    Returns mean time of inversion and number of floating-point operations
    """

    times, flopses = [], []
    for _ in range(tries):
        A = (np.random.sample((2**k, 2**k)) * 2 - 1) * max_value

        # reset flops clock
        inverse.reset_flops()
        # measure time
        begin = time.time()
        B = inverse(A)
        finish = time.time()

        if check:
            assert np.allclose(
                B, np.linalg.inv(A), atol=1e-7
            ), "Check if result of custom inverse function is correct"

        times.append(finish - begin)
        flopses.append(inverse.flops)

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
