# Base model
_base_ = '/home/computador/Desktop/Nexus_Project_3_6/Seal_detection_code/mmyolo/configs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py'

# Dataset root
data_root = '/home/computador/Desktop/TESTON2024DATSET/Seal_detection_TEST_2024.v1-dataset2024.coco/train/'

# Custom classes
class_name = ('Seal', 'Tag_White', 'Tag_Yellow')
num_classes = len(class_name)

# Dataset metadata
metainfo = dict(
    classes=class_name,
    palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
)

# Model config (override for custom num_classes)
model = dict(
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)
    )
)

# ---------------------- Test dataloader ----------------------
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='_annotations.coco.json',   # 👈 your test annotations
        data_prefix=dict(img='./')           # images are in same folder
        # indices=list(range(10))            # 👈 uncomment if only testing few images
    )
)

# ---------------------- Evaluator ----------------------
test_evaluator = dict(ann_file=data_root + '_annotations.coco.json')

# ---------------------- Visualizer ----------------------
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='work_dirs/vis')
    ],
    name='visualizer'
)

# ---------------------- No pretrained here (set via CLI) ----------------------
load_from = None

