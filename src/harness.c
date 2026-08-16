/*!
 * \file harness.c
 * \brief Leave-one-out benchmark for weighted KT coding.
 *
 * Usage: wkt_bench FILE1 FILE2 [FILE3 ...]
 *
 * Each file in turn is the message; the remaining files, concatenated, are the training data. 
 * Reported per file, in bits per symbol:
 *
 *   xi=0     cold-start KT (training discarded);
 *   xi=1     sample-based KT (training at full weight);
 *   xi*      weighted KT at the offline weight: xi* = d/(2*L*Dhat + d) with Dhat the average debiased pairwise divergence
 *            between the training files (the database-spread estimate);
 *   plug-in  weight re-estimated from the message prefix at doubling times;
 *   mixture  twice-universal mixture over the weight grid.
 *
 * Also reported: the message length n, the training length L, the offline divergence estimate Dhat (nats/symbol), and
 * the offline effective training length xi* * L.
 *
 * \copyright Copyright (c) 2026 Yuriy A. Reznik. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wkt.h"

#define MAX_FILES 64

/*! Load a file into a malloc'ed buffer; returns NULL on failure. */
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
    buf = (unsigned char *)malloc((size_t)sz);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (unsigned long)sz;
    return buf;
}

/*! Selected divergence estimator (constant Wilks form by default; --sa selects the support-adaptive variant). */
static double (*div_est)(const unsigned long *, unsigned long, const unsigned long *, unsigned long) = wkt_div_debiased;

static unsigned long msg_cap = 0;

int main(int argc, char **argv)
{
    unsigned long msg_n;
    unsigned char *data[MAX_FILES];
    unsigned long size[MAX_FILES];
    unsigned long cnt[MAX_FILES][WKT_M];
    unsigned long tc[WKT_M];
    unsigned long ltrain;
    double dhat, dsum, xi3, xi4, bpc;
    double l1, l2, l3, l4, l5, l6, xi6;
    double t1, t2, t3, t4, t5, t6;
    unsigned long ntot;
    int nf, i, j, g, npairs, x;

    /* Options: --sa selects the support-adaptive estimator; --cap N truncates each message. */
    if (argc >= 2 && strcmp(argv[1], "--sa") == 0) {
        div_est = wkt_div_debiased_sa;
        argv++;
        argc--;
    }
    if (argc >= 3 && strcmp(argv[1], "--cap") == 0) {
        msg_cap = strtoul(argv[2], NULL, 10);
        argv += 2;
        argc -= 2;
    }
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s [--sa] [--cap N] FILE1 FILE2 [FILE3 ...]\n", argv[0]);
        return 1;
    }
    nf = argc - 1;
    if (nf > MAX_FILES) {
        fprintf(stderr, "too many files (max %d)\n", MAX_FILES);
        return 1;
    }
    /* Load all files and precompute their symbol counts. */
    for (i = 0; i < nf; i++) {
        data[i] = load_file(argv[i + 1], &size[i]);
        if (data[i] == NULL) {
            fprintf(stderr, "cannot read %s\n", argv[i + 1]);
            return 1;
        }
        wkt_counts(data[i], size[i], cnt[i]);
    }

    printf("%-8s %8s %9s %8s %9s %8s %8s %8s %8s %8s %8s %9s %9s\n",
           "file", "n", "L", "Dhat", "leff*", "xi=0", "xi=1",
           "xi*", "plugin", "mixture", "oracle", "xi_fin", "xi_orc");
    t1 = t2 = t3 = t4 = t5 = t6 = 0.0;
    ntot = 0;

    for (i = 0; i < nf; i++) {
        /* Training counts: all files except i. */
        ltrain = 0;
        for (x = 0; x < WKT_M; x++) {
            tc[x] = 0;
        }
        for (g = 0; g < nf; g++) {
            if (g == i) {
                continue;
            }
            ltrain += size[g];
            for (x = 0; x < WKT_M; x++) {
                tc[x] += cnt[g][x];
            }
        }
        /* Offline estimate: average debiased pairwise divergence between the training files (ordered pairs). */
        dsum = 0.0;
        npairs = 0;
        for (g = 0; g < nf; g++) {
            for (j = 0; j < nf; j++) {
                if (g == i || j == i || g == j) {
                    continue;
                }
                dsum += div_est(cnt[g], size[g], cnt[j], size[j]);
                npairs++;
            }
        }
        dhat = npairs > 0 ? dsum / (double)npairs : 0.0;
        xi3 = wkt_xi_star(ltrain, dhat);

        /* Optional cap: truncate the message; the training is unchanged. */
        msg_n = size[i];
        if (msg_cap > 0 && msg_n > msg_cap) {
            msg_n = msg_cap;
        }

        /* Code lengths: xi = 0, xi = 1, offline xi*, plug-in, mixture. */
        l1 = wkt_codelen(data[i], msg_n, tc, ltrain, 0.0);
        l2 = wkt_codelen(data[i], msg_n, tc, ltrain, 1.0);
        l3 = wkt_codelen(data[i], msg_n, tc, ltrain, xi3);
        l4 = wkt_codelen_plugin_est(data[i], msg_n, tc, ltrain,
                                    &xi4, div_est);
        l5 = wkt_codelen_mixture(data[i], msg_n, tc, ltrain);

        /* Oracle: best fixed weight over the grid 2^{-j/4}. */
        l6 = l1;
        xi6 = 0.0;
        for (j = 0; j <= 88; j++) {
            double xw, lw_;
            xw = pow(2.0, -0.25 * (double)j);
            lw_ = wkt_codelen(data[i], msg_n, tc, ltrain, xw);
            if (lw_ < l6) {
                l6 = lw_;
                xi6 = xw;
            }
        }

        bpc = 1.0 / (log(2.0) * (double)msg_n);
        printf("%-8s %8lu %9lu %8.4f %9.1f %8.4f %8.4f %8.4f %8.4f "
               "%8.4f %8.4f %9.2e %9.2e\n",
               argv[i + 1], msg_n, ltrain, dhat,
               xi3 * (double)ltrain,
               l1 * bpc, l2 * bpc, l3 * bpc, l4 * bpc, l5 * bpc,
               l6 * bpc, xi4, xi6);
        t1 += l1; t2 += l2; t3 += l3; t4 += l4; t5 += l5; t6 += l6;
        ntot += msg_n;
    }

    /* Length-weighted per-symbol totals over all messages. */
    bpc = 1.0 / (log(2.0) * (double)ntot);
    printf("%-8s %8lu %9s %8s %9s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n",
           "total", ntot, "-", "-", "-",
           t1 * bpc, t2 * bpc, t3 * bpc, t4 * bpc, t5 * bpc, t6 * bpc);

    for (i = 0; i < nf; i++) {
        free(data[i]);
    }
    return 0;
}
