import json
import numpy as np

class clsConfig:
    """ Class that holds configuration information"""
    
    def __init__(self, cfg_dict):
#         print(type(cfg_dict))
#         print(cfg_dict)
        
        self.camera_id = cfg_dict['Camera_ID']
    
        # dataset root path & file extension
        self.data_root = cfg_dict['DataSource']['source_video']

        # image size (x,y) : tuple
        self.image_size = cfg_dict['DataSource']['image_size']
        
        self.video_fps = cfg_dict['DataSource']['video_fps']

        # Target Object(Tray)의 size (x, y) : tuple
        self.obj_real_size = cfg_dict['DataSource']['obj_real_size']

        # 길이 척도 : pixel값을 mm로 변환하기 위한 factor : list
        self.length_scale = cfg_dict['DataSource']['length_scale']
        
        # detection 동작여부를 판단할 switch file(ON/OFF)
        self.detect_switch = cfg_dict['Detection']['switch_path']
        
        # detection 주기 = video에서 frame을 추출하는 주기 (sec)
        self.detect_interval = cfg_dict['Detection']['interval']
        
        # mmpose model의 config file & checkpoint file
        self.model_config = cfg_dict['DetectionModel']['config']
        self.checkpoint = cfg_dict['DetectionModel']['checkpoint']

        # gpu devices to use for inference
        self.gpu_device = cfg_dict['DetectionModel']['gpu_device']

        # 좌표 변환 matrix : list
        self.mtrx = np.float32(cfg_dict['Calculation']['matrix'])

        # 원본 이미지 기준선 : 변환좌표의 기준선을 역변환 or 정상 이미지의 좌표 사용 : np.array
        self.base_pts0 = np.float32(cfg_dict['Calculation']['base_point'])

        # path to save results & result images
        self.result_path = cfg_dict['Result']['path']
        self.result_image = cfg_dict['Result']['result_image']
        self.result_log = cfg_dict['Result']['result_log']


        
# load configs - AI Camera's basic setting for MSC (in json format)
def load_config(cfg_file='config.json'):
    
    with open(cfg_file, 'r') as f:
        cfg_data = json.load(f)
    
    config = []
    for cfg in cfg_data:
        config.append(clsConfig(cfg))
    
    return config