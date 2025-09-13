_base_ = '/home/computador/Desktop/models/rtmdet_m_custom_seal/rtmdet_m_custom_seal.py'

# ✅ Dataset for testing
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        
        data_root='/home/computador/Desktop/TESTON2024DATSET/Seal_detection_TEST_2024.v1-dataset2024.coco/train/',
        ann_file='_annotations.coco.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=('Seal', 'Tag_White', 'Tag_Yellow'))  # only 3 classes
    )
)

# ✅ Evaluator
test_evaluator = dict(
    
    ann_file='/home/computador/Desktop/TESTON2024DATSET/Seal_detection_TEST_2024.v1-dataset2024.coco/train/_annotations.coco.json',
    metric='bbox'
)

# ✅ Lower test score threshold (so you can actually see low-confidence boxes)
model = dict(
    test_cfg=dict(
        score_thr=0.01  # draw everything >=1% confidence
    )
)

