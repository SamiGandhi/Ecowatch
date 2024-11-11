import os

class parameters:
    captured_video_path = ''
    video_base_path = ''
    captured_frame_dir = ''
    masks_dir = ''    
    reference_frames_dir = ''
    roi_frame_dir = ''
    output_directory = ''
    decoded_frames_dir = ''
    fps = 2
    default_width = 144 #144
    default_height = 144 #144
    gop_coefficient = 30
    quality_factor = 80
    high_quality_factor = 40
    low_quality_factor = 20
    zero = 1
    zone_size = 8
    DCT = "CLA"
    level_numbers = 1
    entropy_coding = "RLE_EG"
    trace_file_path = ""
    packed_load_size = 512
    interleaving = ""
    threshold = 1
    max_level_S = 0
    enhance_method = ''
    enhance_param = ''
    w1 = 8
    w2 = 4
    w3 = 2
    roi_threshold = 10
    running = ''

    POWER = 0.023
    PROC_CLOCK = 7.3728
    CAPTURE_E_PER_BLOCK = 2.65e-6
    CYCLES_PER_DIV = 300
    CYCLES_PER_ADD = 1
    CYCLES_PER_FADD = 20
    CYCLES_PER_SHIFT = 1
    CYCLES_PER_MUL = 2
    CYCLES_PER_FMUL = 100
    CLA_ADD_NB = 464
    CLA_MUL_NB = 192
    MAX_S_LAYERS = 5
    DISTANCE = 100  # meters
    FREQUENCY = 868e6  # Hz (868 MHz is generally more suitable for wildlife monitoring applications) 2.4e9  # Hz (2.4 GHz, common for Wi-Fi)
    ENVIRONEMENT = "wildlife_zone"
    HUMIDITY_LEVEL = 70  # percent
    VEGETATION_DENSITY_LEVEL = 5  # arbitrary unit
    scale_factor = 100
    @classmethod
    def initialize(cls, video_path, runing,DISTANCE,scale_factor,HUMIDITY, VEGETATION,width = 288, hight = 288, gop = 12 ,QF = 60, H = 80,M = 60, L = 20, ENVIRONEMENT = "wildlife_zone" ):
        cls.DISTANCE = DISTANCE
        cls.scale_factor = scale_factor
        cls.captured_video_path = video_path
        cls.video_base_path, _ = os.path.splitext(video_path)
        cls.quality_factor = QF
        cls.running = runing
        cls.default_width = width
        cls.default_height = hight
        cls.high_quality_factor = H
        cls.low_quality_factor = M
        cls.gop_coefficient = gop
        cls.HUMIDITY_LEVEL = HUMIDITY
        cls.VEGETATION_DENSITY_LEVEL = VEGETATION
        cls.zero = L
        cls.ENVIRONEMENT = ENVIRONEMENT
        if(runing == 'Roi' or runing == 'decoding_roi'):
            if(H == 90):
                method = 'ROI/light'
            elif(H == 50):
                method = 'ROI/medium'
            elif(H == 60):
                method = 'ROI/agressive'
            elif(H == 40):
                method = 'ROI/very_agressive'
            else:
                method = 'undefined'
            directory = f"{hight}X{width}/{method}"
            cls.captured_frame_dir = os.path.join(cls.video_base_path, directory, 'captured_frames')
            cls.reference_frames_dir = os.path.join(cls.video_base_path, directory, 'reference_frames')
            cls.roi_frame_dir = os.path.join(cls.video_base_path, directory, 'roi_frames')
            cls.output_directory = os.path.join(cls.video_base_path, directory, 'output')
            cls.decoded_frames_dir = os.path.join(cls.video_base_path, directory, 'decoded')
            cls.trace_file_path = os.path.join(cls.video_base_path, directory, 'trace')
            cls.masks_dir = os.path.join(cls.video_base_path, directory, 'roi_masks')
        else:
            if(QF == 90):
                method = 'MPEG/light'
            elif(QF == 50):
                method = 'MPEG/Medium'
            elif(QF == 40):
                method = 'MPEG/Agressive'
            else:
                method = 'undefined'
            directory = f"{hight}X{width}/{method}/GOP_{cls.gop_coefficient}"
            cls.captured_frame_dir = os.path.join(cls.video_base_path, directory, 'captured_frames')
            cls.reference_frames_dir = os.path.join(cls.video_base_path, directory, 'reference_frames')
            cls.roi_frame_dir = os.path.join(cls.video_base_path, directory, 'roi_frames')
            cls.output_directory = os.path.join(cls.video_base_path, directory, 'output')
            cls.decoded_frames_dir = os.path.join(cls.video_base_path, directory, 'decoded')
            cls.trace_file_path = os.path.join(cls.video_base_path, directory, 'trace')

        
       
        
        
