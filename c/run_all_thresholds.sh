#!/bin/bash

echo "Starting BGE baseline tests for thresholds from 0.75 to 1.00..."
echo "Please ensure you have activated the correct Conda environment before running this script."

# 循环从 xxx 到 xxx，步长为 0.01
for threshold in $(seq 0.90 0.01 0.99)
do
  echo "=========================================================="
  echo "Running test for threshold: $threshold"
  echo "=========================================================="
  
  # 您可以将 CUDA_VISIBLE_DEVICES=3 修改为您想用的任何GPU
  CUDA_VISIBLE_DEVICES=1 python test_baselines_v15.py --method bge --threshold $threshold
  
  # 检查上一个命令是否成功执行，如果失败则中止脚本
  if [ $? -ne 0 ]; then
      echo "Error running test for threshold $threshold. Exiting."
      exit 1
  fi
done

echo "✅ All threshold experiments completed successfully!"