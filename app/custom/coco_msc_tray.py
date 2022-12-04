### mmpose/configs/_base_/datasets/
"""
    0: 'point_1_L1',
    1: 'point_2_L2',
    2: 'point_3_L3',
    3: 'point_4_U1',
    4: 'point_5_U2',
    5: 'point_6_U3'
"""
dataset_info = dict(
    dataset_name='coco_msc_tray',
#     dataset_name='MSCTrayBottomUp',
    paper_info=dict(
        author='SKCC Factory Intelligence',
        title='Tray Keypoints',
        container='AI Camera for MSC(Multi Stacker Crane)',
        year='2022',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='point_1_L1',
            id=1,
            color=[255, 128, 0],
            type='left',
            swap='point_2_L2'),
        1:
        dict(
            name='point_2_L2',
            id=4,
            color=[255, 128, 0],
            type='lower',
            swap='point_5_U2'),
        2:
        dict(
            name='point_3_L3',
            id=1,
            color=[255, 128, 0],
            type='right',
            swap='point_2_L2'),
        3:
        dict(name='point_4_U1',
            id=4, 
            color=[51, 153, 255], 
            type='left', 
            swap='point_5_U2'),        
        4:
        dict(
            name='point_5_U2',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='point_2_L2'),
        5:
        dict(
            name='point_6_U3',
            id=4,
            color=[51, 153, 255],
            type='right',
            swap='point_5_U2')
    },
    skeleton_info={
        0:
        dict(link=('point_1_L1', 'point_2_L2'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('point_2_L2', 'point_3_L3'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('point_4_U1', 'point_5_U2'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('point_5_U2', 'point_6_U3'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('point_4_U1', 'point_1_L1'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('point_5_U2', 'point_2_L2'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('point_6_U3', 'point_3_L3'), id=6, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        1., 1., 1., 1., 1., 1.
    ]
)
