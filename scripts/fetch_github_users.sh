#!/bin/sh
# Fetch the zstd github-users small-record sample set and make the
# deterministic 1-in-18 train/test split used in the paper.
# Requires: curl, zstd, tar.
set -e
curl -sL "https://github.com/facebook/zstd/releases/download/v1.1.3/github_users_sample_set.tar.zst" -o github_users.tar.zst
zstd -dq github_users.tar.zst
tar xf github_users.tar
rm -f github_users.tar github_users.tar.zst
ls github | sort > all.lst
awk 'NR%18==0{print "github/"$0}' all.lst > test.lst
awk 'NR%18!=0{print "github/"$0}' all.lst > train.lst
rm -f all.lst
echo "records in ./github; lists: train.lst test.lst"
echo "run: ./wkt_records train.lst test.lst"
