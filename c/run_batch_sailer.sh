#!/bin/bash

# 设置输出目录名字，和 Python 脚本里保持一致
# 这里的 --method sailer 对应你刚才测通的代码
METHOD="sailer" 
SCRIPT="test_baselines_law.py" # 注意这里改成你刚才改过Bug的文件名

echo "🚀 开始批量测试 $METHOD 的阈值..."

# 循环从 0.0 到 0.9，步长 0.05 (SAILER 是归一化向量，余弦相似度通常在 0-1 之间)
# 如果是 BGE，可能需要在 0.5-0.9 之间测
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