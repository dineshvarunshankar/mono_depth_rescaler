#pragma once
#include <vector>
#include <cmath>

// Minimal dense linear algebra: Gaussian elimination, partial pivoting,
// row-major std::vector<double>.
namespace linalg {

// Solve A X = B in place-free form. A is n x n, B is n x m (row-major).
// Returns false if A is singular. X is n x m.
inline bool solve(std::vector<double> A, std::vector<double> B,
                  int n, int m, std::vector<double>& X) {
    for (int col = 0; col < n; ++col) {
        // partial pivot
        int piv = col;
        double best = std::abs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::abs(A[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-15) return false;
        if (piv != col) {
            for (int c = 0; c < n; ++c) std::swap(A[col * n + c], A[piv * n + c]);
            for (int c = 0; c < m; ++c) std::swap(B[col * m + c], B[piv * m + c]);
        }
        double d = A[col * n + col];
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double f = A[r * n + col] / d;
            if (f == 0.0) continue;
            for (int c = col; c < n; ++c) A[r * n + c] -= f * A[col * n + c];
            for (int c = 0; c < m; ++c) B[r * m + c] -= f * B[col * m + c];
        }
    }
    X.assign(n * m, 0.0);
    for (int r = 0; r < n; ++r) {
        double d = A[r * n + r];
        for (int c = 0; c < m; ++c) X[r * m + c] = B[r * m + c] / d;
    }
    return true;
}

// Solve A x = b for a vector (m = 1).
inline bool solve_vec(const std::vector<double>& A, const std::vector<double>& b,
                      int n, std::vector<double>& x) {
    return solve(A, b, n, 1, x);
}

// Inverse of an n x n matrix (row-major). Returns false if singular.
inline bool inverse(const std::vector<double>& A, int n, std::vector<double>& Ainv) {
    std::vector<double> I(n * n, 0.0);
    for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;
    return solve(A, I, n, n, Ainv);
}

}  // namespace linalg
