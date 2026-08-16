#!/bin/sh
# Fetch a mirror of the Calgary corpus into ./calgary
set -e
curl -sL "https://codeload.github.com/nabcouwer/calgary/tar.gz/refs/heads/master" -o calgary.tar.gz
tar xzf calgary.tar.gz
mv calgary-master calgary
rm calgary.tar.gz
echo "Calgary corpus in ./calgary"
