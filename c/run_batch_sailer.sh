#!/bin/bash

METHOD="sailer" 
SCRIPT="test_baselines_law.py"

echo "🚀 开始批量测试 $METHOD 的阈值..."

# 循环从 xxx 到 xxx，步长 0.01
for threshold in $(seq 0.4 0.01 0.9)
do
  echo "Running threshold: $threshold"
  
  # 这里的 --top_k 建议设大一点（比如 5 或 10），因为阈值过滤是在 top_k 结果里过滤的
  # 如果 top_k 太小，阈值还没起作用就被截断了
  python $SCRIPT --method $METHOD --top_k 10 --threshold $threshold
  
  if [ $? -ne 0 ]; then
      echo "❌ Error at threshold $threshold"
      exit 1
  fi
done

echo "✅ 所有阈值跑完了！"