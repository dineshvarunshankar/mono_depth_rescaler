#include "fits.h"
#include "linalg.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace fits {

// small polynomial helpers (numpy convention: coeffs[0] is the highest power)
static double polyval(const std::vector<double>& c, double x) {
    double y = 0.0;
    for (double ci : c) y = y * x + ci;
    return y;
}

static std::vector<double> polyder(const std::vector<double>& c) {
    const int L = static_cast<int>(c.size());
    if (L <= 1) return {};
    std::vector<double> d(L - 1);
    for (int i = 0; i < L - 1; ++i) d[i] = c[i] * (L - 1 - i);
    return d;
}

// Real roots of a polynomial (Durand-Kerner). Used only for the monotonic
// polynomial's derivative critical points; degrees here are tiny.
static std::vector<double> poly_real_roots(std::vector<double> c) {
    while (c.size() > 1 && c.front() == 0.0) c.erase(c.begin());
    const int deg = static_cast<int>(c.size()) - 1;
    if (deg < 1) return {};
    for (double& ci : c) ci /= c.front();   // monic
    using cd = std::complex<double>;
    std::vector<cd> r(deg);
    cd seed(0.4, 0.9);
    for (int i = 0; i < deg; ++i) r[i] = std::pow(seed, i);
    auto eval = [&](cd z) {
        cd y = 0.0;
        for (double ci : c) y = y * z + ci;
        return y;
    };
    for (int it = 0; it < 100; ++it) {
        double move = 0.0;
        for (int i = 0; i < deg; ++i) {
            cd denom = 1.0;
            for (int j = 0; j < deg; ++j) if (j != i) denom *= (r[i] - r[j]);
            if (std::abs(denom) < 1e-300) continue;
            cd delta = eval(r[i]) / denom;
            r[i] -= delta;
            move = std::max(move, std::abs(delta));
        }
        if (move < 1e-12) break;
    }
    std::vector<double> real;
    for (const cd& z : r)
        if (std::abs(z.imag()) < 1e-6) real.push_back(z.real());
    return real;
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

bool is_valid(const std::function<double(double)>& predict,
              double x_min, double x_max, int n) {
    if (!std::isfinite(x_min) || !std::isfinite(x_max) || x_max <= x_min)
        return false;
    std::vector<double> y(n);
    double ymin = 1e300, ymax = -1e300;
    for (int i = 0; i < n; ++i) {
        double x = x_min + (x_max - x_min) * i / double(n - 1);
        double v = predict(x);
        if (!std::isfinite(v) || v <= 0.0) return false;
        y[i] = v;
        ymin = std::min(ymin, v);
        ymax = std::max(ymax, v);
    }
    // soft monotonicity penalties leave O(scale/kappa) leakage; relative tol
    double tol = -1e-6 * (ymax - ymin + 1e-12);
    for (int i = 1; i < n; ++i)
        if (y[i] - y[i - 1] < tol) return false;
    return true;
}

// weighted least squares for a general design matrix A (m x p, row-major).
// Returns coeffs (p) and covariance sigma^2 (A^T W A)^-1 (p x p, row-major).
static bool wls(const std::vector<double>& A, const std::vector<double>& y,
                const std::vector<double>& w, int m, int p,
                std::vector<double>& coeffs, std::vector<double>& cov) {
    std::vector<double> AtWA(p * p, 0.0), AtWy(p, 0.0);
    for (int i = 0; i < m; ++i) {
        double wi = w[i];
        for (int a = 0; a < p; ++a) {
            double Aa = A[i * p + a];
            AtWy[a] += wi * Aa * y[i];
            for (int b = 0; b < p; ++b)
                AtWA[a * p + b] += wi * Aa * A[i * p + b];
        }
    }
    if (!linalg::solve_vec(AtWA, AtWy, p, coeffs)) return false;

    double sse = 0.0;
    for (int i = 0; i < m; ++i) {
        double pred = 0.0;
        for (int a = 0; a < p; ++a) pred += A[i * p + a] * coeffs[a];
        double r = y[i] - pred;
        sse += w[i] * r * r;
    }
    int dof = std::max(m - p, 1);
    double sigma2 = sse / dof;
    std::vector<double> inv;
    if (linalg::inverse(AtWA, p, inv)) {
        cov.assign(p * p, 0.0);
        for (int i = 0; i < p * p; ++i) cov[i] = sigma2 * inv[i];
    } else {
        cov.clear();
    }
    return true;
}

// polynomial
static Fit fit_polynomial(const std::vector<double>& x, const std::vector<double>& y,
                          const std::vector<double>& w, int degree) {
    const int m = static_cast<int>(x.size());
    const int p = degree + 1;
    std::vector<double> A(m * p);
    for (int i = 0; i < m; ++i) {
        double xi = 1.0;
        for (int j = p - 1; j >= 0; --j) { A[i * p + j] = xi; xi *= x[i]; }
    }
    std::vector<double> coeffs, cov;
    if (!wls(A, y, w, m, p, coeffs, cov)) return {};

    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());
    auto predict = [coeffs](double xx) { return polyval(coeffs, xx); };
    Fit f;
    f.predict = predict;
    f.x_min = x_min; f.x_max = x_max;
    f.valid = is_valid(predict, x_min, x_max);
    f.params = coeffs;
    f.params_cov = cov;
    return f;
}

// Nelder-Mead downhill simplex (scipy defaults: reflect/expand/contract/shrink).
static std::vector<double> nelder_mead(
        const std::function<double(const std::vector<double>&)>& loss,
        std::vector<double> x0) {
    const int n = static_cast<int>(x0.size());
    std::vector<std::vector<double>> s(n + 1, x0);
    for (int i = 0; i < n; ++i) {
        double d = x0[i] != 0.0 ? 0.05 * x0[i] : 0.00025;
        s[i + 1][i] += d;
    }
    std::vector<double> fv(n + 1);
    for (int i = 0; i <= n; ++i) fv[i] = loss(s[i]);
    const int maxiter = 200 * n;
    const double xatol = 1e-4, fatol = 1e-4;

    auto order = [&]() {
        std::vector<int> idx(n + 1);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fv[a] < fv[b]; });
        std::vector<std::vector<double>> ns(n + 1);
        std::vector<double> nf(n + 1);
        for (int i = 0; i <= n; ++i) { ns[i] = s[idx[i]]; nf[i] = fv[idx[i]]; }
        s = ns; fv = nf;
    };

    for (int it = 0; it < maxiter; ++it) {
        order();
        // convergence: simplex spread in x and f both small
        double xspread = 0.0, fspread = 0.0;
        for (int i = 1; i <= n; ++i) {
            fspread = std::max(fspread, std::abs(fv[i] - fv[0]));
            for (int j = 0; j < n; ++j)
                xspread = std::max(xspread, std::abs(s[i][j] - s[0][j]));
        }
        if (xspread <= xatol && fspread <= fatol) break;

        std::vector<double> c(n, 0.0);   // centroid of all but worst
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) c[j] += s[i][j] / n;

        auto comb = [&](double t) {
            std::vector<double> r(n);
            for (int j = 0; j < n; ++j) r[j] = c[j] + t * (c[j] - s[n][j]);
            return r;
        };
        std::vector<double> xr = comb(1.0);   // reflection
        double fr = loss(xr);
        if (fr < fv[0]) {
            std::vector<double> xe = comb(2.0);  // expansion
            double fe = loss(xe);
            if (fe < fr) { s[n] = xe; fv[n] = fe; }
            else         { s[n] = xr; fv[n] = fr; }
        } else if (fr < fv[n - 1]) {
            s[n] = xr; fv[n] = fr;
        } else {
            std::vector<double> xc = comb(0.5);  // contraction
            double fc = loss(xc);
            if (fc < fv[n]) { s[n] = xc; fv[n] = fc; }
            else {                                // shrink
                for (int i = 1; i <= n; ++i) {
                    for (int j = 0; j < n; ++j)
                        s[i][j] = s[0][j] + 0.5 * (s[i][j] - s[0][j]);
                    fv[i] = loss(s[i]);
                }
            }
        }
    }
    order();
    return s[0];
}

static Fit fit_polynomial_monotonic(const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    const std::vector<double>& w, int degree) {
    const double PENALTY = 1e8;
    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());

    Fit init = fit_polynomial(x, y, w, degree);      // WLS start + cov
    if (init.params.empty()) return {};

    auto loss = [&](const std::vector<double>& p) {
        double data = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double r = y[i] - polyval(p, x[i]);
            data += w[i] * r * r;
        }
        std::vector<double> der = polyder(p);
        std::vector<double> der2 = polyder(der);
        std::vector<double> pts = {x_min, x_max};
        for (double root : poly_real_roots(der2))
            if (x_min <= root && root <= x_max) pts.push_back(root);
        double pen = 0.0;
        for (double xp : pts) {
            double dv = polyval(der, xp);
            if (dv < 0.0) pen += dv * dv;
        }
        return data + PENALTY * pen;
    };

    std::vector<double> coeffs = nelder_mead(loss, init.params);
    auto predict = [coeffs](double xx) { return polyval(coeffs, xx); };
    Fit f;
    f.predict = predict;
    f.x_min = x_min; f.x_max = x_max;
    f.valid = is_valid(predict, x_min, x_max);
    f.params = coeffs;
    f.params_cov = init.params_cov;   // unconstrained WLS cov (approximation)
    return f;
}

// exponential:  y = a * exp(b * x),  linearised as log(y) = log(a) + b x
static Fit fit_exponential(const std::vector<double>& x, const std::vector<double>& y,
                           const std::vector<double>& w) {
    std::vector<double> xp, yl, wp;
    for (size_t i = 0; i < y.size(); ++i)
        if (y[i] > 0.0) { xp.push_back(x[i]); yl.push_back(std::log(y[i])); wp.push_back(w[i]); }
    const int m = static_cast<int>(xp.size());
    if (m < 2) return {};

    std::vector<double> A(m * 2);      // columns [x, 1] -> coeffs (b, log_a)
    for (int i = 0; i < m; ++i) { A[i * 2 + 0] = xp[i]; A[i * 2 + 1] = 1.0; }
    std::vector<double> sol, c;        // c: cov in (b, log_a) order
    if (!wls(A, yl, wp, m, 2, sol, c)) return {};
    double b = sol[0], log_a = sol[1], a = std::exp(log_a);
    std::vector<double> params = {a, b};

    // delta method to (a, b) ordering: a = exp(log_a)
    std::vector<double> cov;
    if (c.size() == 4) {
        cov = {a * a * c[3], a * c[2],
               a * c[1],     c[0]};
    }
    double x_min = *std::min_element(xp.begin(), xp.end());
    double x_max = *std::max_element(xp.begin(), xp.end());
    auto predict = [a, b](double xx) { return a * std::exp(b * xx); };
    Fit f;
    f.predict = predict;
    f.x_min = x_min; f.x_max = x_max;
    f.valid = is_valid(predict, x_min, x_max);
    f.params = params;
    f.params_cov = cov;
    return f;
}

// penalised B-spline (P-spline) machinery
static constexpr int SP_DEGREE = 3;
static constexpr int    SP_MAXITER = 30;
static constexpr double SP_RIDGE = 1e-8;

static int find_span(const std::vector<double>& t, int n_ctrl, double x) {
    if (x >= t[n_ctrl]) return n_ctrl - 1;
    if (x <= t[SP_DEGREE]) return SP_DEGREE;
    int lo = SP_DEGREE, hi = n_ctrl;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x < t[mid]) hi = mid; else lo = mid;
    }
    return lo;
}

static void basis_funs(const std::vector<double>& t, int span, double x,
                       double N[SP_DEGREE + 1]) {
    N[0] = 1.0;
    for (int i = 1; i <= SP_DEGREE; ++i) {
        N[i] = 0.0;
    }
    double left[SP_DEGREE + 1] = {};
    double right[SP_DEGREE + 1] = {};
    for (int j = 1; j <= SP_DEGREE; ++j) {
        left[j]  = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double denom = right[r + 1] + left[j - r];
            double temp = denom != 0.0 ? N[r] / denom : 0.0;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

// Build the spline; monotonic toggles the active-set decreasing-coefficient
// penalty. lambda_smoothing is the D3 roughness weight.
static bool build_pspline(std::vector<double> x, std::vector<double> y,
                          std::vector<double> w, int knot_segments,
                          double lambda_smoothing, bool monotonic, double kappa,
                          std::vector<double>& knots_out,
                          std::vector<double>& coeffs_out) {
    // ascending in x
    std::vector<int> ord(x.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return x[a] < x[b]; });
    std::vector<double> xs(x.size()), ys(x.size()), ws(x.size());
    for (size_t i = 0; i < ord.size(); ++i) { xs[i] = x[ord[i]]; ys[i] = y[ord[i]]; ws[i] = w[ord[i]]; }

    const int m = static_cast<int>(xs.size());
    double interval = (xs[m - 1] - xs[0]) / knot_segments;
    if (!(interval > 0.0)) return false;

    const int n_knots = SP_DEGREE * 2 + knot_segments + 1;
    std::vector<double> t(n_knots);
    double a = xs[0] - (SP_DEGREE + 1) * interval;
    double b = xs[m - 1] + (SP_DEGREE + 1) * interval;
    for (int i = 0; i < n_knots; ++i) t[i] = a + (b - a) * i / double(n_knots - 1);
    const int n = n_knots - SP_DEGREE - 1;   // number of basis functions

    // design matrix B (m x n)
    std::vector<double> B(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        int span = find_span(t, n, xs[i]);
        double Nb[SP_DEGREE + 1];
        basis_funs(t, span, xs[i], Nb);
        for (int j = 0; j <= SP_DEGREE; ++j)
            B[i * n + (span - SP_DEGREE + j)] = Nb[j];
    }

    // A = B^T W B + lambda D3^T D3 (+ ridge if lambda == 0); BTy = B^T W y
    std::vector<double> A(n * n, 0.0), BTy(n, 0.0);
    for (int i = 0; i < m; ++i) {
        double wi = ws[i];
        for (int j = 0; j < n; ++j) {
            double Bij = B[i * n + j];
            if (Bij == 0.0) continue;
            BTy[j] += Bij * wi * ys[i];
            for (int l = 0; l < n; ++l)
                A[j * n + l] += Bij * wi * B[i * n + l];
        }
    }
    // D3 third difference: rows i have [-1,3,-3,1] at columns i..i+3
    if (lambda_smoothing > 0.0) {
        static const double d3[4] = {-1, 3, -3, 1};
        for (int i = 0; i + 3 < n; ++i)
            for (int p = 0; p < 4; ++p)
                for (int q = 0; q < 4; ++q)
                    A[(i + p) * n + (i + q)] += lambda_smoothing * d3[p] * d3[q];
    } else {
        for (int i = 0; i < n; ++i) A[i * n + i] += SP_RIDGE;
    }

    // solve (A + D1^T diag(V*kappa) D1) alpha = BTy, iterating the active set
    auto solve_with = [&](const std::vector<int>& V, std::vector<double>& alpha) {
        std::vector<double> M = A;
        for (int e = 0; e + 1 < n; ++e) {
            if (!V[e]) continue;
            double vk = kappa;
            // D1 row e has -1 at e, +1 at e+1
            M[e * n + e]           += vk;
            M[e * n + (e + 1)]     -= vk;
            M[(e + 1) * n + e]     -= vk;
            M[(e + 1) * n + (e + 1)] += vk;
        }
        return linalg::solve_vec(M, BTy, n, alpha);
    };

    std::vector<int> V(n - 1, 0);
    std::vector<double> alpha;
    if (!solve_with(V, alpha)) return false;

    if (monotonic) {
        bool converged = false;
        for (int it = 0; it < SP_MAXITER; ++it) {
            std::vector<int> Vn(n - 1);
            for (int e = 0; e + 1 < n; ++e) Vn[e] = (alpha[e + 1] - alpha[e] < 0.0) ? 1 : 0;
            if (Vn == V) { converged = true; break; }
            V = Vn;
            if (!solve_with(V, alpha)) return false;
        }
        if (!converged) {
            // pin flags cumulatively so the active set only grows
            while (true) {
                std::vector<int> Vn = V;
                for (int e = 0; e + 1 < n; ++e)
                    if (alpha[e + 1] - alpha[e] < 0.0) Vn[e] = 1;
                if (Vn == V) break;
                V = Vn;
                if (!solve_with(V, alpha)) return false;
            }
        }
    }

    knots_out = t;
    coeffs_out = alpha;
    return true;
}

static Fit fit_spline(const std::vector<double>& x, const std::vector<double>& y,
                      const std::vector<double>& w, int knot_segments,
                      double lambda_smoothing, bool monotonic, double kappa) {
    // a cubic B-spline over knot_segments needs knot_segments+degree basis
    // functions; fewer points than that is underdetermined
    if (static_cast<int>(x.size()) < knot_segments + SP_DEGREE) return {};
    std::vector<double> knots, coeffs;
    if (!build_pspline(
            x, y, w, knot_segments, lambda_smoothing, monotonic, kappa,
            knots, coeffs))
        return {};

    const int n_ctrl = static_cast<int>(knots.size()) - SP_DEGREE - 1;
    auto predict = [knots, coeffs, n_ctrl](double xx) -> double {
        double lo = knots[SP_DEGREE], hi = knots[n_ctrl];
        if (xx < lo || xx > hi) return std::nan("");   // extrapolate=False
        int span = find_span(knots, n_ctrl, xx);
        double N[SP_DEGREE + 1];
        basis_funs(knots, span, xx, N);
        double v = 0.0;
        for (int j = 0; j <= SP_DEGREE; ++j) {
            v += coeffs[span - SP_DEGREE + j] * N[j];
        }
        return v;
    };
    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());
    Fit f;
    f.predict = predict;
    f.x_min = x_min; f.x_max = x_max;
    f.valid = is_valid(predict, x_min, x_max);
    // params left empty: spline coeffs vary with knots
    return f;
}

bool is_known_method(const std::string& m) {
    return m == "polynomial" || m == "polynomial_monotonic" || m == "exponential"
        || m == "smoothing_spline" || m == "monotonic_smoothing_spline"
        || m == "monotonic_nonsmoothing_spline";
}

static Fit dispatch(const std::string& method,
                    const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<double>& w, int degree,
                    int num_knots_spline, double spline_kappa) {
    if (method == "polynomial")            return fit_polynomial(x, y, w, degree);
    if (method == "polynomial_monotonic")  return fit_polynomial_monotonic(x, y, w, degree);
    if (method == "exponential")           return fit_exponential(x, y, w);
    if (method == "monotonic_nonsmoothing_spline")
        return fit_spline(x, y, w, num_knots_spline, 0.0, true, spline_kappa);
    if (method == "monotonic_smoothing_spline")
        return fit_spline(x, y, w, num_knots_spline, 1e5, true, spline_kappa);
    if (method == "smoothing_spline")
        // P-spline approximation of scipy's FITPACK UnivariateSpline: not
        // bit-identical, but the deployed monotonic_nonsmoothing_spline is.
        return fit_spline(x, y, w, num_knots_spline, 1e5, false, spline_kappa);
    return {};
}

Fit create(const std::string& method,
           const std::vector<double>& x_in, const std::vector<double>& y_in,
           const std::vector<double>& w_in, int degree, int num_knots_spline,
           double spline_kappa) {
    if (!is_known_method(method)) return {};
    // drop non-finite rows (qVIO uncertainty can be inf/near-zero -> inf weights)
    std::vector<double> x, y, w;
    x.reserve(x_in.size());
    for (size_t i = 0; i < x_in.size(); ++i)
        if (std::isfinite(x_in[i]) && std::isfinite(y_in[i]) && std::isfinite(w_in[i])) {
            x.push_back(x_in[i]); y.push_back(y_in[i]); w.push_back(w_in[i]);
        }
    if (x.size() < 2) return {};
    return dispatch(
        method, x, y, w, degree, num_knots_spline, spline_kappa);
}

Fit create_robust(const std::string& method,
                  const std::vector<double>& x, const std::vector<double>& y,
                  const std::vector<double>& w, int degree, int num_knots_spline,
                  bool outlier_rejection, float outlier_k,
                  double spline_kappa) {
    Fit fit = create(
        method, x, y, w, degree, num_knots_spline, spline_kappa);
    if (!outlier_rejection || !fit.valid) return fit;

    // MAD inliers on the method's own residuals
    const int n = static_cast<int>(x.size());
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = y[i] - fit.predict(x[i]);
    double med = median_of(r);
    std::vector<double> absdev(n);
    for (int i = 0; i < n; ++i) absdev[i] = std::abs(r[i] - med);
    double mad = median_of(absdev) + 1e-9;
    double thr = outlier_k * 1.4826 * mad;

    std::vector<double> xi, yi, wi;
    int kept = 0;
    for (int i = 0; i < n; ++i)
        if (std::abs(r[i] - med) < thr) { xi.push_back(x[i]); yi.push_back(y[i]); wi.push_back(w[i]); ++kept; }

    fit.has_inliers = true;
    fit.inlier_ratio = float(kept) / float(n);
    if (kept < 2 || kept == n) return fit;   // nothing rejected

    Fit refit = create(
        method, xi, yi, wi, degree, num_knots_spline, spline_kappa);
    if (!refit.valid) return fit;
    refit.has_inliers = true;
    refit.inlier_ratio = float(kept) / float(n);
    return refit;
}

std::function<double(double)> rebuild(const std::string& method,
                                      const std::vector<double>& params) {
    if (method == "exponential") {
        double a = params[0], b = params[1];
        return [a, b](double xx) { return a * std::exp(b * xx); };
    }
    // polynomial / polynomial_monotonic
    std::vector<double> c = params;
    return [c](double xx) { return polyval(c, xx); };
}

}  // namespace fits
