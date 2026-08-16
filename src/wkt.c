/*!
 * \file wkt.c
 * \brief Implementation of weighted KT coding with training data.
 * \copyright Copyright (c) 2026 Yuriy A. Reznik. MIT License.
 */

#include <math.h>
#include <stdlib.h>
#include "wkt.h"

/*! Half the alphabet size: total of the Jeffreys 1/2 pseudo-counts. */
#define WKT_M_HALF (0.5 * (double)WKT_M)

void wkt_counts(const unsigned char *seq, unsigned long n, unsigned long *c) {
    unsigned long i;
    for (i = 0; i < (unsigned long)WKT_M; i++) {
        c[i] = 0;
    }
    for (i = 0; i < n; i++) {
        c[seq[i]]++;
    }
}

double wkt_div_plugin(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb) {
    double da, db, pa, pb, d;
    int x;
    da = (double)na + WKT_M_HALF;
    db = (double)nb + WKT_M_HALF;
    d = 0.0;
    for (x = 0; x < WKT_M; x++) {
        pa = ((double)a[x] + 0.5) / da;
        pb = ((double)b[x] + 0.5) / db;
        d += pa * log(pa / pb);
    }
    return d;
}

double wkt_div_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb) {
    double d;
    d = wkt_div_plugin(a, na, b, nb)
        - 0.5 * WKT_D * (1.0 / (double)na + 1.0 / (double)nb);
    return d > 0.0 ? d : 0.0;
}

double wkt_div_debiased_sa(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb) {
    double da, db, pa, pb, ra, rb, va, vb, d, bias;
    int x;
    da = (double)na + 0.5 * (double)WKT_M;
    db = (double)nb + 0.5 * (double)WKT_M;
    d = 0.0;
    bias = 0.0;
    for (x = 0; x < WKT_M; x++) {
        pa = ((double)a[x] + 0.5) / da;
        pb = ((double)b[x] + 0.5) / db;
        d += pa * log(pa / pb);
        ra = (double)a[x] / (double)na;
        rb = (double)b[x] / (double)nb;
        va = ra * (1.0 - ra) / (double)na;
        vb = rb * (1.0 - rb) / (double)nb;
        bias += 0.5 * (va / pa + pa * vb / (pb * pb));
    }
    d -= bias;
    return d > 0.0 ? d : 0.0;
}

double wkt_xi_star(unsigned long ltrain, double dhat)
{
    return WKT_D / (2.0 * (double)ltrain * dhat + WKT_D);
}

double wkt_codelen(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain,
                   double xi) {
    unsigned long k[WKT_M];
    unsigned long t;
    double len, num, den;
    int x;
    for (x = 0; x < WKT_M; x++) {
        k[x] = 0;
    }
    len = 0.0;
    for (t = 0; t < n; t++) {
        x = msg[t];
        num = xi * (double)tc[x] + (double)k[x] + 0.5;
        den = xi * (double)ltrain + (double)t + WKT_M_HALF;
        len -= log(num / den);
        k[x]++;
    }
    return len;
}

double wkt_codelen_plugin_est(const unsigned char *msg, unsigned long n, const unsigned long *tc,
                              unsigned long ltrain, double *xi_out,
                              double (*est)(const unsigned long *, unsigned long,
                                            const unsigned long *, unsigned long)) {
    unsigned long k[WKT_M];
    unsigned long t, next_update;
    double len, num, den, xi, dhat;
    int x;
    for (x = 0; x < WKT_M; x++) {
        k[x] = 0;
    }
    len = 0.0;
    xi = 1.0;
    next_update = 1;
    for (t = 0; t < n; t++) {
        if (t == next_update) {
            dhat = est(tc, ltrain, k, t);
            xi = wkt_xi_star(ltrain, dhat);
            next_update *= 2;
        }
        x = msg[t];
        num = xi * (double)tc[x] + (double)k[x] + 0.5;
        den = xi * (double)ltrain + (double)t + WKT_M_HALF;
        len -= log(num / den);
        k[x]++;
    }
    if (xi_out != NULL) {
        *xi_out = xi;
    }
    return len;
}

double wkt_codelen_plugin(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain,
                          double *xi_out) {
    return wkt_codelen_plugin_est(msg, n, tc, ltrain, xi_out,
                                  wkt_div_debiased);
}

double wkt_codelen_mixture(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain) {
    unsigned long k[WKT_M];
    unsigned long t, v;
    double *xi, *lw, *p;
    double len, wmax, wsum, pmix, q;
    int narms, j, x;
    /* Grid xi_j = 2^{-j}, j = 0..K, K = ceil(log2 L), plus xi = 0. */
    narms = 1;
    v = 1;
    while (v < ltrain) {
        v *= 2;
        narms++;
    }
    narms += 1; /* cold-start arm */
    xi = (double *)malloc((size_t)narms * sizeof(double));
    lw = (double *)malloc((size_t)narms * sizeof(double));
    p  = (double *)malloc((size_t)narms * sizeof(double));
    if (xi == NULL || lw == NULL || p == NULL) {
        free(xi);
        free(lw);
        free(p);
        return -1.0;
    }
    for (j = 0; j < narms - 1; j++) {
        xi[j] = pow(0.5, (double)j);
        lw[j] = 0.0; /* uniform prior: common constant cancels */
    }
    xi[narms - 1] = 0.0;
    lw[narms - 1] = 0.0;
    for (x = 0; x < WKT_M; x++) {
        k[x] = 0;
    }
    len = 0.0;
    for (t = 0; t < n; t++) {
        x = msg[t];
        /* Per-arm predictive probabilities. */
        for (j = 0; j < narms; j++) {
            p[j] = (xi[j] * (double)tc[x] + (double)k[x] + 0.5)
                 / (xi[j] * (double)ltrain + (double)t + WKT_M_HALF);
        }
        /* Posterior-weighted mixture probability. */
        wmax = lw[0];
        for (j = 1; j < narms; j++) {
            if (lw[j] > wmax) {
                wmax = lw[j];
            }
        }
        wsum = 0.0;
        pmix = 0.0;
        for (j = 0; j < narms; j++) {
            q = exp(lw[j] - wmax);
            wsum += q;
            pmix += q * p[j];
        }
        pmix /= wsum;
        len -= log(pmix);
        /* Per-arm likelihood updates. */
        for (j = 0; j < narms; j++) {
            lw[j] += log(p[j]);
        }
        k[x]++;
    }
    free(xi);
    free(lw);
    free(p);
    return len;
}

double wkt_div_mix(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb, double gamma) {
    double da, pa, pb, pm, d;
    int x;
    /* Training side KT-smoothed; message side raw. For gamma > 0 the mixture is bounded below by
       gamma * P~, so the raw empirical needs no smoothing, which avoids the first-order distortion
       of the m/2 pseudo-counts on short records. */
    da = (double)na + WKT_M_HALF;
    d = 0.0;
    for (x = 0; x < WKT_M; x++) {
        pa = ((double)a[x] + 0.5) / da;
        pb = (double)b[x] / (double)nb;
        pm = gamma * pa + (1.0 - gamma) * pb;
        d += pa * log(pa / pm);
    }
    return d;
}

double wkt_div_mix_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb,
                            double gamma) {
    double da, pa, pb, pm, d, corr, g1;
    int x;
    /* Delta-method bias correction: the second-order bias of the plug-in profile is (1-gamma)^2/2 times the Fisher-
       weighted variance of the message-side empirical, sum_x Var(B~(x)) * P~(x) / M~(x)^2, which reduces to (d/2)(1/na
       + 1/nb) on full support at gamma = 0 and adapts to the effective alphabet otherwise. The training-side term is
       included for completeness. */
    da = (double)na + WKT_M_HALF;
    g1 = 1.0 - gamma;
    d = 0.0;
    corr = 0.0;
    for (x = 0; x < WKT_M; x++) {
        pa = ((double)a[x] + 0.5) / da;
        pb = (double)b[x] / (double)nb;
        pm = gamma * pa + (1.0 - gamma) * pb;
        d += pa * log(pa / pm);
        corr += (g1 * g1 * pb * (1.0 - pb) / (double)nb
                 + gamma * gamma * pa * (1.0 - pa) / (double)na)
                * pa / (pm * pm);
    }
    d -= 0.5 * corr;
    return d > 0.0 ? d : 0.0;
}

double wkt_xi_global(unsigned long ltrain, unsigned long n, wkt_dprofile dfun, void *ctx) {
    double lo, hi, g, g1, f, xi;
    int it;
    lo = 1e-12;
    hi = 1.0 - 1e-9;
    for (it = 0; it < 100; it++) {
        g = 0.5 * (lo + hi);
        g1 = 1.0 - g;
        f = g1 / g - (double)n / (double)ltrain
            - (2.0 * (double)n / WKT_D) * dfun(g, ctx) / (g1 * g1);
        if (f > 0.0) {
            lo = g;
        } else {
            hi = g;
        }
    }
    g = 0.5 * (lo + hi);
    xi = ((double)n / (double)ltrain) * g / (1.0 - g);
    if (xi > 1.0) {
        xi = 1.0;
    }
    return xi;
}

/*! Context for the prefix-based divergence profile. */
struct wkt_prefix_ctx {
    const unsigned long *tc;
    unsigned long ltrain;
    const unsigned long *k;
    unsigned long t;
};

/*! Divergence profile from training counts and message-prefix counts. */
static double wkt_prefix_dprofile(double gamma, void *ctx)
{
    const struct wkt_prefix_ctx *c = (const struct wkt_prefix_ctx *)ctx;
    return wkt_div_mix_debiased(c->tc, c->ltrain, c->k, c->t, gamma);
}

double wkt_codelen_plugin_global(const unsigned char *msg, unsigned long n, const unsigned long *tc,
                                 unsigned long ltrain, double *xi_out) {
    unsigned long k[WKT_M];
    unsigned long t, next_update;
    double len, num, den, xi;
    struct wkt_prefix_ctx ctx;
    int x;
    for (x = 0; x < WKT_M; x++) {
        k[x] = 0;
    }
    ctx.tc = tc;
    ctx.ltrain = ltrain;
    ctx.k = k;
    len = 0.0;
    xi = 1.0;
    next_update = 1;
    for (t = 0; t < n; t++) {
        if (t == next_update) {
            ctx.t = t;
            xi = wkt_xi_global(ltrain, n, wkt_prefix_dprofile, &ctx);
            next_update *= 2;
        }
        x = msg[t];
        num = xi * (double)tc[x] + (double)k[x] + 0.5;
        den = xi * (double)ltrain + (double)t + WKT_M_HALF;
        len -= log(num / den);
        k[x]++;
    }
    if (xi_out != NULL) {
        *xi_out = xi;
    }
    return len;
}

double wkt_chi2_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb) {
    double da, pa, pb, chi, corr;
    int x;
    da = (double)na + WKT_M_HALF;
    chi = 0.0;
    corr = 0.0;
    for (x = 0; x < WKT_M; x++) {
        pa = ((double)a[x] + 0.5) / da;
        pb = (double)b[x] / (double)nb;
        chi += (pa - pb) * (pa - pb) / pa;
        corr += pb * (1.0 - pb) / pa;
    }
    if (nb >= 2) {
        chi -= corr / (double)(nb - 1);
    }
    chi *= 0.5;
    return chi > 0.0 ? chi : 0.0;
}

double wkt_xi_harmonic(unsigned long ltrain, double d2)
{
    double leff;
    leff = 1.0 / (1.0 / (double)ltrain + 2.0 * d2 / WKT_D);
    return leff / (double)ltrain;
}

double wkt_codelen_plugin_chi2(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain,
                               double *xi_out) {
    unsigned long k[WKT_M];
    unsigned long t, next_update;
    double len, num, den, xi;
    int x;
    for (x = 0; x < WKT_M; x++) {
        k[x] = 0;
    }
    len = 0.0;
    xi = 1.0;
    next_update = 2;
    for (t = 0; t < n; t++) {
        if (t == next_update) {
            xi = wkt_xi_harmonic(ltrain,
                                 wkt_chi2_debiased(tc, ltrain, k, t));
            next_update *= 2;
        }
        x = msg[t];
        num = xi * (double)tc[x] + (double)k[x] + 0.5;
        den = xi * (double)ltrain + (double)t + WKT_M_HALF;
        len -= log(num / den);
        k[x]++;
    }
    if (xi_out != NULL) {
        *xi_out = xi;
    }
    return len;
}
