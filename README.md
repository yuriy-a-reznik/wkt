# wkt — Weighted KT estimator

Reference implementation accompanying the paper

> Y. A. Reznik, *Optimal Weighting of Training Data in Adaptive Coding*, 2026.

The coding distribution is a Krichevsky–Trofimov (KT) estimator over the
byte alphabet (m = 256, d = 255) whose training counts enter once, scaled
by a weight ξ ∈ [0, 1], and whose message counts enter at full weight:

    Q(x | training, prefix) = (ξ·C[training] + k[prefix] + 1/2) / (ξ·L + t + m/2).

The paper shows that the redundancy-minimizing weight is
ξ\* = d / (2·L·D + d), where L is the training length and D the per-symbol
KL divergence between the training and message sources, equivalently the
harmonic law 1/(ξ\*L) = 1/L + 2D/d for the optimal effective training
length.

## Contents

| File            | Description                                          |
| --------------- | ---------------------------------------------------- |
| `src/wkt.h`     | Library interface (doxygen-documented)               |
| `src/wkt.c`     | Five coders and the divergence estimators            |
| `src/harness.c` | Leave-one-out file benchmark (`wkt_bench`)           |
| `src/records.c` | Small-record benchmark (`wkt_records`)               |
| `Makefile`      | Build (C89, requires only libm)                      |
| `Doxyfile`      | Doxygen configuration                                |

## The five coders

1. **ξ = 0** — cold-start KT, training data discarded;
2. **ξ = 1** — sample-based KT, training data at full weight;
3. **ξ = ξ\*** — weighted KT at an externally supplied weight, here from
   the offline database estimate: the average debiased pairwise
   divergence between the training files (paper, Sec. 5.1);
4. **plug-in** — the weight is re-estimated from the message prefix at
   doubling times t = 1, 2, 4, 8, … and fed back (paper, Sec. 5.2); the
   schedule is a function of decoded data only, so the decoder
   reproduces it exactly;
5. **mixture** — twice-universal uniform mixture over the weight grid
   ξ_j = 2^(−j), j = 0..⌈log₂ L⌉, plus a cold-start arm (paper,
   Sec. 5.3); within ln(K+2) nats of the best arm, uniformly in D.

The harness additionally reports the **oracle**: the best fixed weight,
found by a sweep over the grid 2^(−j/4).

All code lengths are ideal arithmetic-code lengths, −ln Q, reported in
bits per symbol.

## Build and run

    make
    ./wkt_bench FILE1 FILE2 [FILE3 ...]

The ten Calgary/Canterbury test files used in the paper are included
under `corpus/`, so the benchmark runs with no downloads:

    ./wkt_bench corpus/bib corpus/book1 corpus/book2 corpus/news \
        corpus/paper1 corpus/paper2 corpus/alice29.txt \
        corpus/asyoulik.txt corpus/lcet10.txt corpus/plrabn12.txt

Each file in turn is treated as the message; the remaining files,
concatenated, form the training data. Example, on the English subset of
the Calgary corpus:

    ./scripts/fetch_texts.sh
    ./wkt_bench bib book1 book2 news paper1 paper2 \
        alice29.txt asyoulik.txt lcet10.txt plrabn12.txt

reproduces the tables of the paper: the ten English texts of the
Calgary and Canterbury corpora (`--sa` selects the support-adaptive
`--cap N` caps each message at its first N bytes (training unchanged), for small-message experiments.
form of the bias correction in place of the constant Wilks form).
Further benchmarks bracket the regime: the full Canterbury corpus
(`scripts/fetch_canterbury.sh`, binary-dominated training, oracle at
xi=0), the split-book Calgary variant (`scripts/fetch_calgary.sh`,
`scripts/split_books.sh`), and a small-record stream benchmark on the
zstd `github-users` sample set (nearly homogeneous records, oracle
near xi=1):

    ./scripts/fetch_github_users.sh
    ./wkt_records train.lst test.lst The small-record
benchmark of Section 7 runs on the Zstandard github-users sample set
(`scripts/fetch_github_users.sh`):

    ./wkt_records ghusers/github

Each file in the directory is one record; every 18th record (sorted by
name) is a test record and the rest, concatenated, are the training
data. `wkt_records` additionally exercises the global-law solver
(`wkt_xi_global`), the divergence-profile estimator
(`wkt_div_mix_debiased`), and the record-stream plug-in.

## License

MIT. See `LICENSE`.
