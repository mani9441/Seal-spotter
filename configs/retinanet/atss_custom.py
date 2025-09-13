_base_ = '/home/computador/Desktop/models/retina_custom/atss_custom.py'

# ✅ Override dataset for testing
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root='/home/computador/Desktop/TESTON2024DATSET/Seal_detection_TEST_2024.v1-dataset2024.coco/train/',
        ann_file='_annotations.coco.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=('Seal', 'Tag_White', 'Tag_Yellow'))  # drop "objects"
    )
)

# ✅ Evaluator
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/computador/Desktop/TESTON2024DATSET/Seal_detection_TEST_2024.v1-dataset2024.coco/train/_annotations.coco.json',
    metric='bbox'
)

# ✅ Lower threshold so you can *see* predictions even if AP is low
model = dict(
    test_cfg=dict(
        score_thr=0.01   # draw boxes >= 0.01 confidence
    )
)

