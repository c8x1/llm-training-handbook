#!/bin/bash
# 下载本教程使用的所有数据集
set -e

echo "Downloading datasets..."

# tiny-shakespeare
mkdir -p data/shakespeare_char
wget -q -O data/shakespeare_char/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "  tiny-shakespeare: done"

# Alpaca dataset (SFT)
wget -q -O data/alpaca_data.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
echo "  alpaca: done"

echo "All datasets downloaded to data/"
