/*!
 * \file wkt.c
 * \brief Weighted Krichevsky-Trofimov (KT) coding with training data: implementation.
 *
 *  Reference implementation accompanying the paper 
 *    "Optimal Weighting of Training Data in Adaptive Coding" (Y. A. Reznik, 2026). 
 * 
 * The coding distribution is a KT estimator whose training counts enter once, scaled by a weight xi in [0,1],
 * and whose message counts enter at full weight:
 *
 *   Q(x | training, prefix) = (xi*C[x] + k[x] + 1/2) / (xi*L + t + m/2),
 *
 * where C[x] are training counts, L their total, k[x] message-prefix counts, t their total, and m the alphabet
 * size. The theory gives the redundancy-minimizing weight xi* = d/(2*L*D + d), with d = m-1 and D the
 * per-symbol Kullback-Leibler divergence between the training and message sources, and the harmonic law
 * 1/(xi*L) = 1/L + 2D/d for the optimal effective training length.
 *
 * The module implements:
 *
 *   counts        wkt_counts                 symbol counts of a sequence;
 *   estimators    wkt_div_plugin             plug-in KL divergence of KT-smoothed empiricals;
 *                 wkt_div_debiased           Wilks-debiased divergence (constant correction);
 *                 wkt_div_debiased_sa        support-adaptive (delta-method) debiased divergence;
 *                 wkt_div_mix                divergence profile D(P~ || M~_gamma) of the global law;
 *                 wkt_div_mix_debiased       its delta-method-debiased form;
 *                 wkt_chi2_debiased          debiased chi-square (quadratic) divergence;
 *   weights       wkt_xi_star                two-scale optimal weight xi* = d/(2*L*D + d);
 *                 wkt_xi_global              global-law weight, solved by bisection;
 *                 wkt_xi_harmonic            second-order global weight via the harmonic law;
 *   coders        wkt_codelen                weighted KT at a fixed weight (xi = 0, 1, or given);
 *                 wkt_codelen_plugin         plug-in feedback loop, weight reset at doubling times;
 *                 wkt_codelen_plugin_est     same, with a caller-selected divergence estimator;
 *                 wkt_codelen_plugin_global  plug-in loop driven by the global law;
 *                 wkt_codelen_plugin_chi2    plug-in loop driven by the chi-square estimator;
 *                 wkt_codelen_mixture        twice-universal mixture over a weight grid.
 *
 * All schemes are sequentially decodable: every quantity used by the encoder is computable from the training
 * sequence and the already decoded message prefix.
 *
 * \copyright Copyright (c) 2026 Yuriy A. Reznik. MIT License.
 * \date 2026-06-01
 */

#include <math.h>
#include <stdlib.h>
#include "wkt.h"

/*! Half the alphabet size: total of the Jeffreys 1/2 pseudo-counts. */
#define WKT_M_HALF (0.5 * (double)WKT_M)

/*!
 * \brief Symbol counts of a sequence.
 * \param seq  Input sequence.
 * \param n    Sequence length in bytes.
 * \param c    Output array of WKT_M counts; overwritten.
 */
void wkt_counts(const unsigned char *seq, unsigned long n, unsigned long *c)
{
    unsigned long i;
    /* Zero the table, then tally the sequence. */
    for (i = 0; i < (unsigned long)WKT_M; i++) {
        c[i] = 0;
    }
    for (i = 0; i < n; i++) {
        c[seq[i]]++;
    }
}

/*!
 * \brief Plug-in KL divergence between KT-smoothed empiricals, in nats.
 *
 * Computes D(P~ || Q~) with P~[x] = (a[x]+1/2)/(na+m/2) and likewise Q~ from (b, nb). Smoothing keeps the value finite
 * when one sample contains symbols the other lacks.
 *
 * \param a   Counts of the first sample.
 * \param na  Size of the first sample.
 * \param b   Counts of the second sample.
 * \param nb  Size of the second sample.
 * \return    D(P~ || Q~) in nats per symbol.
 */
double wkt_div_plugin(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb)
{
    double da, db, pa, pb, d;
    int x;
    /* KT-smoothed empiricals; accumulate sum_x P~(x) log(P~(x)/Q~(x)). */
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

/*!
 * \brief Debiased divergence estimate, in nats.
 *
 * Subtracts the second-order bias of the plug-in divergence between independent samples, (d/2)*(1/na + 1/nb) -- the
 * Wilks excess of the two empirical fits -- and clamps the result at zero:
 *
 *   Dhat = max(0, D(P~ || Q~) - (d/2)*(1/na + 1/nb)).
 *
 * \param a   Counts of the first sample.
 * \param na  Size of the first sample.
 * \param b   Counts of the second sample.
 * \param nb  Size of the second sample.
 * \return    Debiased estimate of D(P || Q) in nats per symbol.
 */
double wkt_div_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb)
{
    double d;
    /* Subtract the Wilks bias of the two fits; clamp at zero. */
    d = wkt_div_plugin(a, na, b, nb)
        - 0.5 * WKT_D * (1.0 / (double)na + 1.0 / (double)nb);
    return d > 0.0 ? d : 0.0;
}

/*!
 * \brief Support-adaptive (delta-method) debiased divergence, in nats.
 *
 * Same target as wkt_div_debiased, but the bias correction uses the empirical variances instead of the full-alphabet
 * Wilks constant:
 *
 *   bias = (1/2) * sum_x [ va(x)/Pa(x) + Pa(x)*vb(x)/Pb(x)^2 ],
 *
 * with va(x) = A(x)(1-A(x))/na from the raw frequencies A(x)=a[x]/na, and likewise vb. Symbols absent from both samples
 * contribute nothing, so the correction adapts to the occupied sub-alphabet; on full support it agrees with the
 * (d/2)(1/na+1/nb) constant of wkt_div_debiased at second order.
 *
 * \param a   Counts of the first sample.
 * \param na  Size of the first sample.
 * \param b   Counts of the second sample.
 * \param nb  Size of the second sample.
 * \return    Debiased estimate of D(P || Q) in nats per symbol.
 */
double wkt_div_debiased_sa(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb)
{
    double da, db, pa, pb, ra, rb, va, vb, d, bias;
    int x;
    da = (double)na + 0.5 * (double)WKT_M;
    db = (double)nb + 0.5 * (double)WKT_M;
    d = 0.0;
    bias = 0.0;
    /* One pass: plug-in divergence and delta-method bias from the empirical variances. */
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

/*!
 * \brief Optimal weight from the harmonic law.
 *
 * Returns xi* = d/(2*L*D + d), the redundancy-minimizing weight for a training sequence of length L at per-symbol
 * divergence D from the message source.
 *
 * \param ltrain  Training length L.
 * \param dhat    Divergence estimate D, in nats per symbol.
 * \return        Weight xi* in (0, 1].
 */
double wkt_xi_star(unsigned long ltrain, double dhat)
{
    return WKT_D / (2.0 * (double)ltrain * dhat + WKT_D);
}

/*!
 * \brief Ideal code length of the weighted KT coder at a fixed weight.
 *
 * Encodes the message with the KT estimator initialized by the training counts scaled by xi. xi = 0 gives the
 * cold-start KT coder; xi = 1 gives the sample-based coder.
 *
 * \param msg     Message sequence.
 * \param n       Message length.
 * \param tc      Training counts (array of WKT_M).
 * \param ltrain  Training length (sum of tc).
 * \param xi      Weight in [0, 1].
 * \return        Code length -ln Q in nats.
 */
double wkt_codelen(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain,
                   double xi)
{
    unsigned long k[WKT_M];
    unsigned long t;
    double len, num, den;
    int x;
    for (x = 0; x < WKT_M; x++) {
        k[x] = 0;
    }
    len = 0.0;
    /* One pass: predictive probability of the weighted KT estimator, code-length increment, count update. */
    for (t = 0; t < n; t++) {
        x = msg[t];
        num = xi * (double)tc[x] + (double)k[x] + 0.5;
        den = xi * (double)ltrain + (double)t + WKT_M_HALF;
        len -= log(num / den);
        k[x]++;
    }
    return len;
}

/*!
 * \brief As wkt_codelen_plugin, with a caller-selected divergence estimator (wkt_div_debiased or wkt_div_debiased_sa).
 */
double wkt_codelen_plugin_est(const unsigned char *msg, unsigned long n, const unsigned long *tc,
                              unsigned long ltrain, double *xi_out,
                              double (*est)(const unsigned long *, unsigned long,
                                            const unsigned long *, unsigned long))
{
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
        /* At doubling times, re-estimate the mismatch from the prefix and reset the weight. */
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
    /* Report the final (converged) weight. */
    if (xi_out != NULL) {
        *xi_out = xi;
    }
    return len;
}

/*!
 * \brief Ideal code length of the plug-in feedback loop.
 *
 * Starts at xi = 1 and, at message times t = 1, 2, 4, 8, ..., re-estimates the mismatch by the debiased divergence
 * between the training empirical and the message-prefix empirical, and resets xi_t = d/(2*L*Dhat_t + d). The weight is
 * constant between updates. The decoder reproduces the schedule exactly, since it is a function of the training
 * sequence and the decoded prefix.
 *
 * \param msg     Message sequence.
 * \param n       Message length.
 * \param tc      Training counts (array of WKT_M).
 * \param ltrain  Training length.
 * \param xi_out  If non-NULL, receives the final weight used.
 * \return        Code length -ln Q in nats.
 */
double wkt_codelen_plugin(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain,
                          double *xi_out)
{
    return wkt_codelen_plugin_est(msg, n, tc, ltrain, xi_out,
                                  wkt_div_debiased);
}

/*!
 * \brief Ideal code length of the twice-universal mixture over weights.
 *
 * Encodes with the uniform mixture of weighted KT coders over the geometric grid xi_j = 2^{-j}, j = 0..K, K = ceil(log2
 * L), plus the cold-start arm xi = 0. The mixture code length exceeds that of the best arm by at most ln(K+2) nats,
 * uniformly in the mismatch.
 *
 * \param msg     Message sequence.
 * \param n       Message length.
 * \param tc      Training counts (array of WKT_M).
 * \param ltrain  Training length.
 * \return        Code length -ln Q_mix in nats.
 */
double wkt_codelen_mixture(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain)
{
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
            if (lw[j] > wmax)
{
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

/*!
 * \brief Divergence profile term of the global law: D(P~ || M~_gamma).
 *
 * Computes D(P~ || gamma*P~ + (1-gamma)*Q~) with KT-smoothed empiricals P~ from (a, na) and Q~ from (b, nb): the
 * divergence of the training empirical from the pseudo-count mixture at skew gamma. At gamma = 0 this is
 * wkt_div_plugin.
 *
 * \param a      Counts of the training-side sample.
 * \param na     Its size.
 * \param b      Counts of the message-side sample.
 * \param nb     Its size.
 * \param gamma  Skew in [0, 1).
 * \return       D(P~ || M~_gamma) in nats per symbol.
 */
double wkt_div_mix(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb, double gamma)
{
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

/*!
 * \brief Debiased divergence profile term.
 *
 * Subtracts the second-order bias of wkt_div_mix between independent samples, (1-gamma)^2 * (d/2) * (1/na + 1/nb), and
 * clamps at zero. At gamma = 0 this reduces to wkt_div_debiased.
 */
double wkt_div_mix_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb,
                            double gamma)
{
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

/*!
 * \brief Optimal weight from the global law (paper, Corollary 4.3).
 *
 * Solves, by bisection in gamma on (0, 1), the scalar equation
 *
 *   (1-gamma)/gamma = n/L + (2n/d) * D(gamma) / (1-gamma)^2,
 *
 * and returns xi* = (n/L) * gamma* / (1 - gamma*), clipped to [0, 1]. Requires the message length n, which is known to
 * both the encoder and the decoder in record- and packet-oriented applications. With D identically zero the solution is
 * xi* = 1; as n/L -> infinity it reduces to the two-scale law xi* = d/(2*L*D + d).
 *
 * \param ltrain  Training length L.
 * \param n       Message length.
 * \param dfun    Divergence profile D(gamma), e.g. built on wkt_div_mix_debiased.
 * \param ctx     Context passed to dfun.
 * \return        Weight xi* in [0, 1].
 */
double wkt_xi_global(unsigned long ltrain, unsigned long n, wkt_dprofile dfun, void *ctx)
{
    double lo, hi, g, g1, f, xi;
    int it;
    /* Bisection on the skew gamma: f is decreasing, with a sign change on (0, 1). */
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
    /* Map the solved skew back to the weight; clip at 1. */
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

/*!
 * \brief Plug-in feedback loop driven by the global law.
 *
 * As wkt_codelen_plugin, but at each doubling time the weight is reset by solving the global law with the divergence
 * profile estimated from the training counts and the message-prefix counts, wkt_div_mix_debiased. Sequentially
 * decodable for known n.
 *
 * \param msg     Message sequence.
 * \param n       Message length.
 * \param tc      Training counts (array of WKT_M).
 * \param ltrain  Training length.
 * \param xi_out  If non-NULL, receives the final weight used.
 * \return        Code length -ln Q in nats.
 */
double wkt_codelen_plugin_global(const unsigned char *msg, unsigned long n, const unsigned long *tc,
                                 unsigned long ltrain, double *xi_out)
{
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
        /* At doubling times, reset the weight by solving the global law on the current prefix. */
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

/*!
 * \brief Debiased chi-square divergence estimate, in nats.
 *
 * Estimates D2 = (1/2) * sum_x (P(x)-Q(x))^2 / P(x), the quadratic divergence of the per-symbol law (paper, Theorem
 * 4.4), which agrees with the KL divergence to second order in the mismatch. The denominator is the KT-smoothed
 * training empirical (well estimated); the message side is the raw empirical, with its multinomial variance removed
 * exactly:
 *
 *   D2hat = max(0, (1/2)[ sum (P~-Q^)^2/P~
 *                        - (1/(nb-1)) sum Q^(1-Q^)/P~ ]).
 *
 * Unlike the log plug-in, this estimator has no missing-mass inflation on short samples, which makes it the estimator
 * of choice when the message side has few symbols.
 *
 * \param a   Training-side counts.
 * \param na  Training-side size.
 * \param b   Message-side counts.
 * \param nb  Message-side size (must be >= 2).
 * \return    Debiased estimate of D2 in nats per symbol.
 */
double wkt_chi2_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb)
{
    double da, pa, pb, chi, corr;
    int x;
    /* Smoothed training side, raw message side; the exact multinomial variance is removed below. */
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

/*!
 * \brief Weight from the second-order global law.
 *
 * At second order in the mismatch the global law (paper, Corollary 4.3) is independent of the message length and
 * reduces to the harmonic law in the effective training length,
 *
 *   1/leff* = 1/L + 2*D2/d,   xi = leff/L,
 *
 * the per-symbol law of Theorem 4.4. This is the practical form for the small-record regime.
 *
 * \param ltrain  Training length L.
 * \param d2      Quadratic divergence estimate.
 * \return        Weight xi* in (0, 1].
 */
double wkt_xi_harmonic(unsigned long ltrain, double d2)
{
    double leff;
    /* Harmonic law: reciprocal effective lengths add. */
    leff = 1.0 / (1.0 / (double)ltrain + 2.0 * d2 / WKT_D);
    return leff / (double)ltrain;
}

/*!
 * \brief Plug-in feedback loop driven by the second-order global law.
 *
 * As wkt_codelen_plugin, but the mismatch is estimated by wkt_chi2_debiased and the weight is set by wkt_xi_harmonic at
 * each doubling time. Suited to short messages, where the chi-square estimator avoids the missing-mass inflation of the
 * log plug-in.
 *
 * \param msg     Message sequence.
 * \param n       Message length.
 * \param tc      Training counts (array of WKT_M).
 * \param ltrain  Training length.
 * \param xi_out  If non-NULL, receives the final weight used.
 * \return        Code length -ln Q in nats.
 */
double wkt_codelen_plugin_chi2(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain,
                               double *xi_out)
{
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
        /* At doubling times, reset the weight from the chi-square mismatch estimate. */
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
