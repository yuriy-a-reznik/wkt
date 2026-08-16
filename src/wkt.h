/*!
 * \file wkt.h
 * \brief Weighted Krichevsky-Trofimov (KT) coding with training data: public interface.
 *
 * Reference implementation accompanying the paper 
 *    "Optimal Weighting of Training Data in Adaptive Coding" (Y. A. Reznik, 2026). 
 * 
 * Full documentation of each function accompanies its implementation in wkt.c.
 *
 * \copyright Copyright (c) 2026 Yuriy A. Reznik. MIT License.
 * \date 2026-06-01
 */

#ifndef WKT_H
#define WKT_H

/*! Alphabet size: byte-oriented sources. */
#define WKT_M 256

/*! Model dimension d = m - 1. */
#define WKT_D 255.0

/*! Symbol counts of a sequence. */
void wkt_counts(const unsigned char *seq, unsigned long n, unsigned long *c);

/*! Plug-in KL divergence between KT-smoothed empiricals, in nats. */
double wkt_div_plugin(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

/*! Debiased divergence estimate, in nats. */
double wkt_div_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

/*! Support-adaptive (delta-method) debiased divergence, in nats. */
double wkt_div_debiased_sa(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

/*! Optimal weight from the harmonic law. */
double wkt_xi_star(unsigned long ltrain, double dhat);

/*! Ideal code length of the weighted KT coder at a fixed weight. */
double wkt_codelen(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain, double xi);

/*! Ideal code length of the plug-in feedback loop. */
double wkt_codelen_plugin(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain, double *xi_out);

/*! As wkt_codelen_plugin, with a caller-selected divergence estimator (wkt_div_debiased or wkt_div_debiased_sa). */
double wkt_codelen_plugin_est(const unsigned char *msg, unsigned long n, const unsigned long *tc,
                              unsigned long ltrain, double *xi_out,
                              double (*est)(const unsigned long *, unsigned long, const unsigned long *, unsigned long));

/*! Ideal code length of the twice-universal mixture over weights. */
double wkt_codelen_mixture(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain);


/*! Divergence profile term of the global law: D(P~ || M~_gamma). */
double wkt_div_mix(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb, double gamma);

/*! Debiased divergence profile term. */
double wkt_div_mix_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb, double gamma);

/*! Divergence-profile callback for the global-law solver. */
typedef double (*wkt_dprofile)(double gamma, void *ctx);

/*! Optimal weight from the global law (paper, Corollary 4.3). */
double wkt_xi_global(unsigned long ltrain, unsigned long n, wkt_dprofile dfun, void *ctx);

/*! Plug-in feedback loop driven by the global law. */
double wkt_codelen_plugin_global(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain, double *xi_out);

/*! Debiased chi-square divergence estimate, in nats. */
double wkt_chi2_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

/*! Weight from the second-order global law. */
double wkt_xi_harmonic(unsigned long ltrain, double d2);

/*! Plug-in feedback loop driven by the second-order global law. */
double wkt_codelen_plugin_chi2(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain, double *xi_out);

#endif /* WKT_H */
