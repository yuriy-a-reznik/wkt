#!/bin/sh
# Split Calgary book1 and book2 into consecutive 128 KiB parts
# (book1.1..book1.6, book2.1..book2.5), as used in the paper.
set -e
for f in book1 book2; do
    i=1
    off=0
    sz=$(wc -c < "$f")
    while [ "$off" -lt "$sz" ]; do
        dd if="$f" of="$f.$i" bs=131072 skip=$((off / 131072)) count=1 2>/dev/null
        off=$((off + 131072))
        i=$((i + 1))
    done
done
echo "done"
