from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'polyp'
        self.subset = 'kvasir'
        self.data_root = '/path/to/your/dataset'
        self.use_test_set = True
        self.num_channel = 3  # Number of input image channels
        self.num_class = 1    # Number of output classes
        self.base_channel = 32  # Base number of channels
        # Model
        self.model = 'unet'
        self.base_channel = 32
        self.model_path = r'C:\Users\HP\Downloads\best.pth'  # Full path to your trained model
        # Training
        self.total_epoch = 400
        self.train_bs = 16
        self.loss_type = 'ce'
        self.optimizer_type = 'adam'

        # Validating
        self.metrics = ['dice', 'iou']
        self.val_bs = 1

        # Training setting
        self.use_ema = False
        self.logger_name = 'medseg_trainer'

        # Augmentation
        self.crop_size = 320
        self.randscale = [-0.5, 1.0]
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5
        self.v_flip = 0.5
