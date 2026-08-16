#!/bin/sh
# Fetch the text/code members of the Canterbury corpus from the
# zlib-ng/corpora mirror (newline-normalized variants) and split the
# files exceeding 128 KiB into consecutive 128 KiB parts, as used in
# the paper. Requires: curl.
set -e
BASE="https://raw.githubusercontent.com/zlib-ng/corpora/master/canterbury"
for f in alice29.txt asyoulik.txt cp.html fields.c grammar.lsp \
         lcet10.txt plrabn12.txt xargs.1; do
    curl -sL "$BASE/$f" -o "$f"
done
for f in alice29 lcet10 plrabn12; do
    i=1
    off=0
    sz=$(wc -c < "$f.txt")
    while [ "$off" -lt "$sz" ]; do
        dd if="$f.txt" of="$f.$i" bs=131072 skip=$((off / 131072)) \
           count=1 2>/dev/null
        off=$((off + 131072))
        i=$((i + 1))
    done
    rm "$f.txt"
done
echo "done; run:"
echo "  ./wkt_bench cp.html fields.c grammar.lsp xargs.1 asyoulik.txt \\"
echo "      alice29.1 alice29.2 lcet10.1 lcet10.2 lcet10.3 lcet10.4 \\"
echo "      plrabn12.1 plrabn12.2 plrabn12.3 plrabn12.4"
