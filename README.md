
# MMDetection on VOC2007 with Mask R-CNN & Sparse R-CNN

本项目基于 OpenMMLab 的 MMDetection 框架，完成了 VOC2007 数据集上的目标检测任务，训练模型包括 Mask R-CNN 和 Sparse R-CNN，支持训练、测试、推理与可视化。

## 1. 环境准备

### 创建 Conda 虚拟环境
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### 安装 PyTorch（GPU，CUDA 12.1）
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
```

### 安装 MMDetection 及依赖
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

git clone git@github.com:open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

## 2. 数据准备

### 下载并解压 VOC2007 数据集
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

### 转换为 COCO 格式
```bash
python tools/dataset_converters/pascal_voc.py mmdetection/data/VOCdevkit     --out-format coco -o mmdetection/data/coco
```

## 3. 配置文件修改

- `configs/_base_/models/mask-rcnn_r50_fpn.py`：设置 `num_classes=20`
- `configs/_base_/datasets/coco_instance.py`：设置数据路径
- `configs/_base_/schedules/schedule_1x.py`：添加 `max_epochs=48`
- `mmdet/datasets/xml_style.py`：相对路径改为绝对路径


## 4. 模型训练

### Mask R-CNN 训练
```bash
export CUDA_VISIBLE_DEVICES=6
python tools/train.py mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py     --work-dir work_dirs3/mask_rcnn_voc
```


### Sparse R-CNN 多卡训练
```bash
export CUDA_VISIBLE_DEVICES=5,6,7
bash tools/dist_train.sh configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py 3     --work-dir work_dirs3/sparse_rcnn_voc
```

## 5. TensorBoard 可视化

```bash
tensorboard --logdir mmdetection/work_dirs3/mask_rcnn_voc/20250526_140843/vis_data --port 6007

tensorboard --logdir mmdetection/work_dirs/sparse_rcnn_voc/20250527_200710/vis_data --port 6008
```

## 6. 可视化 Proposal 与最终检测结果

```bash
python comparison/visualize_comparison.py   --config configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py   --checkpoint work_dirs3/mask_rcnn_voc/epoch_48.pth   --output-dir comparison/proposal_results
```

## 7. 模型测试

### Mask R-CNN
```bash
python tools/test.py     work_dirs3/mask_rcnn_voc/mask-rcnn_r50_fpn_1x_coco.py     work_dirs3/mask_rcnn_voc/epoch_48.pth     --show-dir comparison/mask/
```

### Sparse R-CNN
```bash
python tools/test.py     configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py     work_dirs3/sparse_rcnn_voc/epoch_60.pth     --show-dir comparison/sparse/
```

## 8. 测试自定义图片

### Mask R-CNN
```bash
python demo/image_demo.py     comparison/demo_image     configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py     --weights work_dirs/mask_rcnn_voc/epoch_48.pth     --out-dir comparison/my_test/     --device cuda
```

### Sparse R-CNN
```bash
python demo/image_demo.py     comparison/demo_image     configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py     --weights work_dirs/sparse_rcnn_voc/epoch_60.pth     --out-dir comparison/my_test_sparse/     --device cuda
```

---

## 项目目录结构建议

```
mmdetection/
├── configs/
├── tools/
├── work_dirs3/
│   ├── mask_rcnn_voc/
│   └── sparse_rcnn_voc/
├── comparison/
│   ├── proposal_results/
│   ├── mask/
│   ├── sparse/
│   └── demo_image/
│   └── my_test_sparse/
│   └── my_test_mask/
└── data/
    └── coco/
```

## 参考资源

- [MMDetection 文档](https://github.com/open-mmlab/mmdetection)
- [Pascal VOC 官网](http://host.robots.ox.ac.uk/pascal/VOC/)
- [OpenMMLab 官网](https://openmmlab.com/)
