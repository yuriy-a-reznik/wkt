/*!
 * \file records.c
 * \brief Small-record benchmark in the global (non-vanishing skew) regime: a stream of records against a fixed training set.
 *
 * Usage: wkt_records TRAINLIST TESTLIST
 *
 * TRAINLIST and TESTLIST are text files with one path per line. The training records, concatenated, form the training
 * data. Each test record is its own message (per-record KT state). Coders compared, reported as aggregate bits per
 * symbol and bits per record:
 *
 *   xi=0        cold-start KT per record;
 *   xi=1        sample-based KT per record;
 *   offline-2s  weighted KT at the two-scale offline weight, xi* = d/(2*L*Dhat + d), Dhat the average debiased
 *               divergence of each training record from the rest;
 *   offline-gl  weighted KT at the global-law weight: the scalar equation of the paper's Corollary "global law" solved
 *               by bisection, with the divergence profile D(P_T||M_g) estimated by the delta-method-debiased plug-in over
 *               held-out training records;
 *   plugin-rec  plug-in loop restarted within each record (updates at doubling times of the record prefix);
 *   plugin-str  plug-in over the record stream: the weight for each record is set from the accumulated counts of the
 *               previously encoded records, constant within a record;
 *   mixture     twice-universal mixture over the weight grid, run independently per record;
 *   oracle-rec  best fixed weight per record (sweep 2^{-j/4}), summed; the mean best weight is also reported.
 *
 * All schemes except the oracle are sequentially decodable from the training data and the decoded stream.
 *
 * \copyright Copyright (c) 2026 Yuriy A. Reznik. MIT License.
 * \date 2026-06-01
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wkt.h"

#define MAX_RECORDS 20000
#define MAX_PATH    512
#define NSWEEP      89

/*!
 * \brief Load a file into memory.
 * \param path  Path to the file.
 * \param n     Output size of the file in bytes.
 * \return      Pointer to the allocated buffer, or NULL on failure.
 */
static unsigned char *load_file(const char *path, unsigned long *n)
{
    FILE *f;
    unsigned char *buf;
    long sz;
    f = fopen(path, "rb");
    if (f == NULL) {
        return NULL;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    buf = (unsigned char *)malloc((size_t)(sz > 0 ? sz : 1));
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (unsigned long)sz;
    return buf;
}

/*!
 * \brief Load a list of file paths from a text file.
 * \param path   Path to the list file.
 * \param names  Output array of strings to hold the file paths.
 * \return       Number of paths loaded, or -1 on failure.
 */
static int load_list(const char *path, char (*names)[MAX_PATH])
{
    FILE *f;
    int r;
    char line[MAX_PATH];
    f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    r = 0;
    while (fgets(line, MAX_PATH, f) != NULL && r < MAX_RECORDS) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len > 0) {
            strcpy(names[r], line);
            r++;
        }
    }
    fclose(f);
    return r;
}

/*!
 * Delta-method-debiased estimate of D(P_T || M_gamma) at skew g: training side smoothed (hc over lh), message side the
 * raw record empirical (rc over m_len). Admissible at raw message empiricals because the mixture is bounded below by g
 * * P~.
 */
static double profile_debiased(const unsigned long *hc, unsigned long lh, const unsigned long *rc, unsigned long m_len, double g)
{
    double dp, bias, pt, bb, mg, vb, vp;
    int x;
    dp = 0.0;
    bias = 0.0;
    for (x = 0; x < WKT_M; x++) {
        pt = ((double)hc[x] + 0.5) / ((double)lh + 0.5 * (double)WKT_M);
        bb = (double)rc[x] / (double)m_len;
        mg = g * pt + (1.0 - g) * bb;
        dp += pt * log(pt / mg);
        vb = bb * (1.0 - bb) / (double)m_len;
        vp = pt * (1.0 - pt) / (double)lh;
        bias += 0.5 * pt
              * ((1.0 - g) * (1.0 - g) * vb + g * g * vp) / (mg * mg);
    }
    dp -= bias;
    return dp > 0.0 ? dp : 0.0;
}

/*!
 * \brief Main entry point for the small-record benchmark.
 * \param argc  Argument count.
 * \param argv  Argument vector: TRAINLIST TESTLIST.
 * \return      Exit code: 0 on success, non-zero on failure.
 */
int main(int argc, char **argv)
{
    static char trn_names[MAX_RECORDS][MAX_PATH];
    static char tst_names[MAX_RECORDS][MAX_PATH];
    static unsigned long tc[WKT_M];
    static unsigned long hc[WKT_M];
    static unsigned long gc[WKT_M];
    unsigned long *trn_cnt; /* per-record training counts */
    unsigned long *trn_len;
    unsigned long ltrain, ntot, gt;
    double dsum, dhat, xi2s, xigl, gam, bpc, bpr;
    double t0, t1, t2s, tgl, tpr, tps, tmx, tor;
    double sweep_xi[NSWEEP], orc_xi_sum, xistr;
    unsigned long nrec;
    int ntrn, ntst, i, j, x;

    if (argc != 3) {
        fprintf(stderr, "usage: %s TRAINLIST TESTLIST\n", argv[0]);
        return 1;
    }
    ntrn = load_list(argv[1], trn_names);
    ntst = load_list(argv[2], tst_names);
    if (ntrn <= 1 || ntst <= 0) {
        fprintf(stderr, "bad lists (%d train, %d test)\n", ntrn, ntst);
        return 1;
    }
    trn_cnt = (unsigned long *)malloc((size_t)ntrn * WKT_M
                                      * sizeof(unsigned long));
    trn_len = (unsigned long *)malloc((size_t)ntrn
                                      * sizeof(unsigned long));
    if (trn_cnt == NULL || trn_len == NULL) {
        return 1;
    }

    /* Pass over training: total and per-record counts. */
    for (x = 0; x < WKT_M; x++) {
        tc[x] = 0;
    }
    ltrain = 0;
    for (i = 0; i < ntrn; i++) {
        unsigned long n;
        unsigned char *b = load_file(trn_names[i], &n);
        if (b == NULL) {
            fprintf(stderr, "cannot read %s\n", trn_names[i]);
            return 1;
        }
        wkt_counts(b, n, trn_cnt + (size_t)i * WKT_M);
        for (x = 0; x < WKT_M; x++) {
            tc[x] += trn_cnt[(size_t)i * WKT_M + x];
        }
        ltrain += n;
        trn_len[i] = n;
        free(b);
    }

    /* Offline two-scale estimate: average debiased divergence of the rest-of-training from each training record. */
    dsum = 0.0;
    for (i = 0; i < ntrn; i++) {
        for (x = 0; x < WKT_M; x++) {
            hc[x] = tc[x] - trn_cnt[(size_t)i * WKT_M + x];
        }
        dsum += wkt_div_debiased(hc, ltrain - trn_len[i],
                                 trn_cnt + (size_t)i * WKT_M, trn_len[i]);
    }
    dhat = dsum / (double)ntrn;
    xi2s = wkt_xi_star(ltrain, dhat);

    /* Mean test record length, needed by the global solve. */
    ntot = 0;
    nrec = 0;
    for (i = 0; i < ntst; i++) {
        unsigned long n;
        unsigned char *b = load_file(tst_names[i], &n);
        if (b == NULL) {
            fprintf(stderr, "cannot read %s\n", tst_names[i]);
            return 1;
        }
        ntot += n;
        nrec++;
        free(b);
    }

    /* Offline global solve: bisection on the skew for (1-g)/g = n/L + (2n/d) * Dprof(g)/(1-g)^2, with Dprof(g) the
       profile estimate averaged over held-out training records. */
    {
        double glo = 1e-6, ghi = 1.0 - 1e-6, gmid, lhs, rhs, dprof;
        double nbar = (double)ntot / (double)nrec;
        int it, r;
        for (it = 0; it < 50; it++) {
            gmid = 0.5 * (glo + ghi);
            dprof = 0.0;
            for (r = 0; r < ntrn; r++) {
                for (x = 0; x < WKT_M; x++) {
                    hc[x] = tc[x] - trn_cnt[(size_t)r * WKT_M + x];
                }
                dprof += profile_debiased(hc, ltrain - trn_len[r],
                                          trn_cnt + (size_t)r * WKT_M,
                                          trn_len[r], gmid);
            }
            dprof /= (double)ntrn;
            lhs = (1.0 - gmid) / gmid;
            rhs = nbar / (double)ltrain
                + (2.0 * nbar / WKT_D) * dprof
                  / ((1.0 - gmid) * (1.0 - gmid));
            if (lhs > rhs) {
                glo = gmid;
            } else {
                ghi = gmid;
            }
        }
        gam = 0.5 * (glo + ghi);
        xigl = gam * nbar / ((1.0 - gam) * (double)ltrain);
        if (xigl > 1.0) {
            xigl = 1.0;
        }
    }

    for (j = 0; j < NSWEEP; j++) {
        sweep_xi[j] = pow(2.0, -0.25 * (double)j);
    }

    /* Pass over the test stream. */
    for (x = 0; x < WKT_M; x++) {
        gc[x] = 0;
    }
    gt = 0;
    xistr = 1.0;
    t0 = t1 = t2s = tgl = tpr = tps = tmx = tor = 0.0;
    orc_xi_sum = 0.0;

    for (i = 0; i < ntst; i++) {
        unsigned long n, t;
        double best, best_xi, lenj;
        unsigned char *b = load_file(tst_names[i], &n);
        if (b == NULL) {
            return 1;
        }
        t0 += wkt_codelen(b, n, tc, ltrain, 0.0);
        t1 += wkt_codelen(b, n, tc, ltrain, 1.0);
        t2s += wkt_codelen(b, n, tc, ltrain, xi2s);
        tgl += wkt_codelen(b, n, tc, ltrain, xigl);
        tpr += wkt_codelen_plugin(b, n, tc, ltrain, NULL);
        tmx += wkt_codelen_mixture(b, n, tc, ltrain);

        /* stream plug-in: weight from previously encoded records */
        if (gt > 0) {
            xistr = wkt_xi_star(ltrain,
                                wkt_div_debiased(tc, ltrain, gc, gt));
        }
        tps += wkt_codelen(b, n, tc, ltrain, xistr);
        for (t = 0; t < n; t++) {
            gc[b[t]]++;
        }
        gt += n;

        /* per-record oracle */
        best = wkt_codelen(b, n, tc, ltrain, 0.0);
        best_xi = 0.0;
        for (j = 0; j < NSWEEP; j++) {
            lenj = wkt_codelen(b, n, tc, ltrain, sweep_xi[j]);
            if (lenj < best) {
                best = lenj;
                best_xi = sweep_xi[j];
            }
        }
        tor += best;
        orc_xi_sum += best_xi;
        free(b);
    }

	/* Report the results. */
    bpc = 1.0 / (log(2.0) * (double)ntot);
    bpr = 1.0 / (log(2.0) * (double)nrec);
    printf("train records: %d, L = %lu\n", ntrn, ltrain);
    printf("test records:  %lu, n_total = %lu, n_mean = %.1f\n",
           nrec, ntot, (double)ntot / (double)nrec);
    printf("offline two-scale: Dhat = %.4f, xi = %.3e (leff = %.1f)\n",
           dhat, xi2s, xi2s * (double)ltrain);
    printf("offline global:    gamma = %.4f, xi = %.3e (leff = %.1f)\n",
           gam, xigl, xigl * (double)ltrain);
    printf("stream plug-in final xi = %.3e (leff = %.1f)\n",
           xistr, xistr * (double)ltrain);
    printf("oracle mean xi = %.3f\n", orc_xi_sum / (double)nrec);
    printf("%-12s %12s %12s\n", "method", "bits/symbol", "bits/record");
    printf("%-12s %12.4f %12.1f\n", "xi=0", t0 * bpc, t0 * bpr);
    printf("%-12s %12.4f %12.1f\n", "xi=1", t1 * bpc, t1 * bpr);
    printf("%-12s %12.4f %12.1f\n", "offline-2s", t2s * bpc, t2s * bpr);
    printf("%-12s %12.4f %12.1f\n", "offline-gl", tgl * bpc, tgl * bpr);
    printf("%-12s %12.4f %12.1f\n", "plugin-rec", tpr * bpc, tpr * bpr);
    printf("%-12s %12.4f %12.1f\n", "plugin-str", tps * bpc, tps * bpr);
    printf("%-12s %12.4f %12.1f\n", "mixture", tmx * bpc, tmx * bpr);
    printf("%-12s %12.4f %12.1f\n", "oracle-rec", tor * bpc, tor * bpr);

    free(trn_cnt);
    free(trn_len);
    return 0;
}
