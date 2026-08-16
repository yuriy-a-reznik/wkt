/*!
 * \file wkt.h
 * \brief Weighted Krichevsky-Trofimov (KT) coding with training data.
 *
 * Reference implementation accompanying the paper "Optimal Weighting of Training Data in Universal Source Coding" (Y.
 * A. Reznik, 2026).
 *
 * The coding distribution is a KT estimator whose training counts enter once, scaled by a weight xi in [0,1], and whose
 * message counts enter at full weight:
 *
 *   Q(x | training, prefix) =
 *     (xi*C[x] + k[x] + 1/2) / (xi*L + t + m/2),
 *
 * where C[x] are training counts, L their total, k[x] message-prefix counts, t their total, and m the alphabet size.
 * The theory gives the redundancy-minimizing weight xi* = d/(2*L*D + d), with d = m-1 and D the per-symbol
 * Kullback-Leibler divergence between the training and message sources, and the harmonic law 1/(xi*L) = 1/L + 2D/d for
 * the optimal effective training length.
 *
 * Five coders are provided:
 *   1. wkt_codelen with xi = 0   : cold-start KT (training discarded);
 *   2. wkt_codelen with xi = 1   : sample-based KT (training at full
 *                                  weight);
 *   3. wkt_codelen with xi given : weighted KT, weight supplied by the
 *                                  caller (e.g., from a design-stage
 *                                  divergence estimate);
 *   4. wkt_codelen_plugin        : weight re-estimated from the message
 *                                  prefix at doubling times (plug-in
 *                                  feedback loop);
 *   5. wkt_codelen_mixture       : uniform mixture over a geometric
 *                                  grid of weights (twice-universal
 *                                  code).
 *
 * All code lengths are ideal arithmetic-code lengths, -ln Q, in nats. All schemes are sequentially decodable: every
 * quantity used by the encoder is computable from the training sequence and the already decoded message prefix.
 *
 * The implementation is C89 and has no dependencies beyond libm.
 *
 * \copyright Copyright (c) 2026 Yuriy A. Reznik. MIT License.
 */

#ifndef WKT_H
#define WKT_H

/*! Alphabet size: byte-oriented sources. */
#define WKT_M 256

/*! Model dimension d = m - 1. */
#define WKT_D 255.0

/*!
 * \brief Symbol counts of a sequence.
 * \param seq  Input sequence.
 * \param n    Sequence length in bytes.
 * \param c    Output array of WKT_M counts; overwritten.
 */
void wkt_counts(const unsigned char *seq, unsigned long n, unsigned long *c);

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
double wkt_div_plugin(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

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
double wkt_div_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

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
double wkt_div_debiased_sa(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

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
double wkt_xi_star(unsigned long ltrain, double dhat);

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
double wkt_codelen(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain, double xi);

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
                          double *xi_out);

/*!
 * \brief As wkt_codelen_plugin, with a caller-selected divergence estimator (wkt_div_debiased or wkt_div_debiased_sa).
 */
double wkt_codelen_plugin_est(const unsigned char *msg, unsigned long n, const unsigned long *tc,
                              unsigned long ltrain, double *xi_out,
                              double (*est)(const unsigned long *, unsigned long,
                                            const unsigned long *, unsigned long));

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
double wkt_codelen_mixture(const unsigned char *msg, unsigned long n, const unsigned long *tc, unsigned long ltrain);


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
double wkt_div_mix(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb, double gamma);

/*!
 * \brief Debiased divergence profile term.
 *
 * Subtracts the second-order bias of wkt_div_mix between independent samples, (1-gamma)^2 * (d/2) * (1/na + 1/nb), and
 * clamps at zero. At gamma = 0 this reduces to wkt_div_debiased.
 */
double wkt_div_mix_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb,
                            double gamma);

/*!
 * \brief Divergence-profile callback for the global-law solver.
 * \param gamma  Skew in [0, 1).
 * \param ctx    Caller context.
 * \return       Estimate of D(P_T || M_gamma) in nats per symbol.
 */
typedef double (*wkt_dprofile)(double gamma, void *ctx);

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
double wkt_xi_global(unsigned long ltrain, unsigned long n, wkt_dprofile dfun, void *ctx);

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
                                 unsigned long ltrain, double *xi_out);

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
double wkt_chi2_debiased(const unsigned long *a, unsigned long na, const unsigned long *b, unsigned long nb);

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
double wkt_xi_harmonic(unsigned long ltrain, double d2);

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
                               double *xi_out);

#endif /* WKT_H */
