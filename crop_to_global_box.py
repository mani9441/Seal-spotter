import json
import torch
import numpy as np

from mmcv.transforms import BaseTransform
from mmyolo.registry import TRANSFORMS

from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from mmdet.structures.bbox import HorizontalBoxes


@TRANSFORMS.register_module()
class CropToGlobalBox(BaseTransform):
    def __init__(self, ann_file):
        self.ann_file = ann_file
        self.roi = self._compute_global_bbox()

    def _compute_global_bbox(self):
        with open(self.ann_file, 'r') as f:
            data = json.load(f)

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
        for ann in data['annotations']:
            x, y, w, h = ann['bbox']
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        return [int(min_x), int(min_y), int(max_x), int(max_y)]

    def transform(self, results):
        min_x, min_y, max_x, max_y = self.roi
        img = results['img']
        img = img[min_y:max_y, min_x:max_x, :]

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        bboxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']

        keep = []
        new_bboxes = []

        for i, box in enumerate(bboxes.numpy()):
            x1, y1, x2, y2 = box
            new_x1 = max(x1 - min_x, 0)
            new_y1 = max(y1 - min_y, 0)
            new_x2 = min(x2 - min_x, max_x - min_x)
            new_y2 = min(y2 - min_y, max_y - min_y)

            if new_x2 > new_x1 and new_y2 > new_y1:
                new_bboxes.append([new_x1, new_y1, new_x2, new_y2])
                keep.append(i)

        if not new_bboxes:
            return None  # skip empty

        results['gt_bboxes'] = torch.tensor(new_bboxes)
        results['gt_bboxes_labels'] = torch.tensor([labels[i] for i in keep])

        instances = InstanceData()
        instances.bboxes = HorizontalBoxes(results['gt_bboxes'])
        instances.labels = results['gt_bboxes_labels']

        data_sample = DetDataSample()
        data_sample.gt_instances = instances

        results['data_samples'] = data_sample

        # ✅ Convert to Tensor for MMYOLO (important)
        if isinstance(results['img'], np.ndarray):
            results['img'] = torch.from_numpy(results['img']).permute(2, 0, 1).float()

        results['inputs'] = results['img']  # required by yolov5_collate

        return results

