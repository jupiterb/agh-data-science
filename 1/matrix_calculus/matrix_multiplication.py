import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

from matrix_utils import FlopsCounter, get_submatrices


# Multiply class


class Multiply(FlopsCounter):
    def __init__(self, l: int) -> None:
        super().__init__()
        self._l = l

    def __call__(self, A: np.ndarray, B: np.ndarray):
        return self._multiply(A, B)

    def _multiply(self, A: np.ndarray, B: np.ndarray):
        """
        Returns the result of the matrix multiplication
        """
        assert (
            A.shape[1] == B.shape[0]
        ), "number of columns of first matrix should be equal to number of rows of second matrix"

        max_size = max(max(A.shape), max(B.shape))

        return (
            self._normal_multiply(A, B)
            if max_size <= 2**self._l
            else self._bineta_multiply(A, B)
        )

    def _normal_multiply(self, A: np.ndarray, B: np.ndarray):
        """
        Returns the result of the matrix multiplication
        """
        # result matrix
        C = np.zeros((A.shape[0], B.shape[1]))

        # iterational matrices multiply
        for i, row in enumerate(A):
            for j, col in enumerate(B.T):
                for a, b in zip(row, col):
                    C[i, j] += a * b
                    self._flops += 2

        return C

    def _bineta_multiply(self, A: np.ndarray, B: np.ndarray):
        """
        Returns the result of the matrix multiplication
        """
        assert A.shape == B.shape and A.shape[0] == A.shape[1]
        n = A.shape[0]

        if n == 1:
            self._flops += 1
            return A * B

        A11, A12, A21, A22 = get_submatrices(A)
        B11, B12, B21, B22 = get_submatrices(B)

        # multiply submatrices and merge to final matrix
        def make_result_submatrix(
            rows_matrices: tuple[np.ndarray, np.ndarray],
            cols_matrices: tuple[np.ndarray, np.ndarray],
        ):
            # multiply
            M1 = self._multiply(rows_matrices[0], cols_matrices[0])
            M2 = self._multiply(rows_matrices[1], cols_matrices[1])
            result_submatrix = M1 + M2
            # cost of M1 + M2
            self._flops += M1.shape[0] * M1.shape[1]
            return result_submatrix

        C11 = make_result_submatrix((A11, A12), (B11, B21))
        C12 = make_result_submatrix((A11, A12), (B12, B22))
        C21 = make_result_submatrix((A21, A22), (B11, B21))
        C22 = make_result_submatrix((A21, A22), (B12, B22))

        C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

        return C


# Test functions


def run_tests(k_max: int, check=False):
    """
    Plots grid of graphs with measurement of time or floating-point operations of multiplication
    for diffirent k and l parameters (2^k x 2^k is size of matrices, l is limit of recurrent multiplication)
    """
    measurements = {
        l: _measurement_for_l(k_max, l, check=check) for l in range(3, k_max)
    }

    _, axs = plt.subplots(1, 2, figsize=(20, 20))

    for l, measurement in measurements.items():
        times = [m[0] for m in measurement.values()]
        flops = [m[1] for m in measurement.values()]

        axs[0].plot(measurement.keys(), times, label=f"l = {l}")
        axs[1].plot(measurement.keys(), flops, label=f"l = {l}")

    axs[0].set_title("Times [s] for different k (matrix shape is 2^k x 2^k)")
    axs[0].grid()
    axs[0].legend(loc="best")

    axs[1].set_title("Flops for different k (matrix shape is 2^k x 2^k)")
    axs[1].grid()
    axs[1].legend(loc="best")

    plt.show()


def _measurement_for_l(k_max: int, l: int, tries_per_k=2, check=False):
    """
    Returns for each k from [2, k_max] measurement of time and floating-point operations of multiplication
    """
    return {
        k: _multiplication_measurement(k, l, tries_per_k, check)
        for k in range(2, k_max + 1)
    }


def _multiplication_measurement(k: int, l: int, tries=1, max_value=10, check=False):
    """
    For each try generate two matrices with values from -max_value to + max_value
    and sizes 2^k x 2^k, then multiply them.
    If check = True, compare result of multiplication with result of numpy multiply operator
    Returns mean time of mutiplication and number of floating-point operations
    """
    print(f"Test run: k = {k}, l = {l}, tires = {tries}")

    times, flopses = [], []
    multiply = Multiply(l)

    for _ in range(tries):
        A = (np.random.sample((2**k, 2**k)) * 2 - 1) * max_value
        B = (np.random.sample((2**k, 2**k)) * 2 - 1) * max_value

        # reset flops clock
        multiply.reset_flops()
        # measure time
        begin = time.time()
        C = multiply(A, B)
        finish = time.time()

        if check:
            assert np.allclose(
                C, A @ B, atol=1e-8
            ), "Check if result of custom mutiply function is correct"

        times.append(finish - begin)
        flopses.append(multiply.flops)

    return np.mean(times), np.mean(flopses)


# Main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--check", type=bool, default=False)

    args = parser.parse_args()
    k = args.k
    check = args.check

    run_tests(k, check)
