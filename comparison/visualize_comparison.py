import os
import os.path as osp
import random
import torch
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np

from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS
from mmdet.apis import init_detector


def draw_boxes(img, boxes, color, label):
    img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file path', required=True)
    parser.add_argument('--checkpoint', help='Checkpoint file', required=True)
    parser.add_argument('--output-dir', help='Directory to save visualizations', required=True)
    parser.add_argument('--num-samples', type=int, default=4, help='Number of images to visualize')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    device = 'cuda:0'  # 你用的GPU设备，或者'cpu'

    # Build model and load checkpoint
    model = init_detector(cfg, args.checkpoint, device=device)
    model.eval()

    # Load dataset
    test_dataset_cfg = cfg.test_dataloader.dataset
    test_dataset_cfg.test_mode = True
    dataset = DATASETS.build(test_dataset_cfg)

    indices = random.sample(range(len(dataset)), args.num_samples)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        data = dataset[idx]
        data_batch = default_collate([data])

        # 通过模型自带的 data_preprocessor 预处理
        data_batch = model.data_preprocessor(data_batch)

        # 将数据移动到正确设备
        def move_to_device(data, device):
            if isinstance(data, torch.Tensor):
                return data.to(device)
            elif isinstance(data, dict):
                return {k: move_to_device(v, device) for k, v in data.items()}
            elif isinstance(data, list):
                return [move_to_device(x, device) for x in data]
            else:
                return data

        data_batch = move_to_device(data_batch, device)

        with torch.no_grad():
            result = model.test_step(data_batch)[0]

            feats = model.extract_feat(data_batch['inputs'][0].unsqueeze(0))
            # 打印 rpn_head 的所有方法名（调试用）
            print("Available methods in rpn_head:", dir(model.rpn_head))
            proposal_list = model.rpn_head.predict(
                feats, data_batch['data_samples'], rescale=False
            )


        # 获取图像路径（新版）
        print(data['data_samples'])
        print(data['data_samples'].metainfo)
        img_meta = data['data_samples'].metainfo
        img_path = img_meta['img_path']
        print("Loading image from:", img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        proposals = proposal_list[0].bboxes[:500].cpu().numpy()  # 提取前500个框并转为 numpy
        proposal_img = draw_boxes(img, proposals, (0, 255, 0), 'Proposals')
        # bboxes = result.pred_instances.bboxes.cpu().numpy()
        # print(f"[{i}] Number of predicted boxes: {len(bboxes)}")
        # print(f"[{i}] First few predicted boxes:\n{bboxes[:5]}")
        # print(result.pred_instances)
        # pred_img = draw_boxes(img, bboxes, (255, 0, 0), 'Predictions')

        # vis_img = np.concatenate((proposal_img, pred_img), axis=1)
        out_path = osp.join(args.output_dir, f'vis_{i}.jpg')
        plt.imsave(out_path, proposal_img)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()



