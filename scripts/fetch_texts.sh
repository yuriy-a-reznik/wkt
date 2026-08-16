#!/bin/sh
# Fetch the ten English-text files of the Calgary and Canterbury
# corpora used in the paper: six Calgary texts and four Canterbury
# texts, from GitHub mirrors (Canterbury texts newline-normalized).
# Requires: curl, tar.
set -e
curl -sL "https://codeload.github.com/nabcouwer/calgary/tar.gz/refs/heads/master" -o calgary.tar.gz
tar xzf calgary.tar.gz
for f in bib book1 book2 news paper1 paper2; do
    mv "calgary-master/$f" .
done
rm -rf calgary-master calgary.tar.gz
BASE="https://raw.githubusercontent.com/zlib-ng/corpora/master/canterbury"
for f in alice29.txt asyoulik.txt lcet10.txt plrabn12.txt; do
    curl -sL "$BASE/$f" -o "$f"
done
echo "done; run:"
echo "  ./wkt_bench bib book1 book2 news paper1 paper2 \\"
echo "      alice29.txt asyoulik.txt lcet10.txt plrabn12.txt"
