#!/bin/bash

# 脚本用于依次运行ogbl-biokg数据集的不同extractor实验

echo "========================================="
echo "开始运行ogbl-biokg数据集的实验"
echo "========================================="

# 实验1: randomWalk
echo ""
echo "[1/3] 运行 randomWalk extractor..."
echo "----------------------------------------"
python main.py --dataset ogbl-biokg --extractor randomWalk
if [ $? -ne 0 ]; then
    echo "错误: randomWalk 实验失败"
    exit 1
fi
echo "✓ randomWalk 实验完成"

# 实验2: khop-subtree
echo ""
echo "[2/3] 运行 khop-subtree extractor..."
echo "----------------------------------------"
python main.py --dataset ogbl-biokg --extractor khop-subtree
if [ $? -ne 0 ]; then
    echo "错误: khop-subtree 实验失败"
    exit 1
fi
echo "✓ khop-subtree 实验完成"

# 实验3: probability
echo ""
echo "[3/3] 运行 probability extractor..."
echo "----------------------------------------"
python main.py --dataset ogbl-biokg --extractor probability
if [ $? -ne 0 ]; then
    echo "错误: probability 实验失败"
    exit 1
fi
echo "✓ probability 实验完成"

echo ""
echo "========================================="
echo "所有ogbl-biokg实验已完成！"
echo "========================================="
