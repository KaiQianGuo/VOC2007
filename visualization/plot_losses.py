import json
import pandas as pd
import matplotlib.pyplot as plt

def load_logs(log_path):
    """加载单个模型的JSON日志"""
    with open(log_path, 'r') as f:
        logs = [json.loads(line) for line in f if '"loss"' in line]
    return pd.DataFrame({
        'step': [log['step'] for log in logs],
        'loss': [log['loss'] for log in logs]
    })

# 加载两个模型的日志
mask_df = load_logs(
    '/data/disk2/guokaiqian/mmdetection/work_dirs3/mask_rcnn_voc/20250526_140843/vis_data/20250526_140843.json'
)
sparse_df = load_logs(
    '/data/disk2/guokaiqian/mmdetection/work_dirs1/sparse_rcnn_voc/20250522_192743/vis_data/20250522_192743.json'
)

# 创建对比图
plt.figure(figsize=(14, 7))
plt.plot(mask_df['step'], mask_df['loss'], label='Mask R-CNN', linewidth=1.5)
plt.plot(sparse_df['step'], sparse_df['loss'], label='Sparse R-CNN', linewidth=1.5)

plt.title('Training Loss Comparison (Mask vs Sparse R-CNN)', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Total Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存到对比目录
plt.savefig('/data/disk2/guokaiqian/mmdetection/comparison/loss_comparison_final.png', 
           dpi=350, bbox_inches='tight')