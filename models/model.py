
class ModelGAN(object):
    """
        Basic model for later model implementations
        GAN models = Generative Adversarial Networks

        TODO:
         - Support for all models
        """
    def __init__(self, input_width, input_height, input_channels, input_glob_channels, batch_size=32):
        """
        Init

        :param input_width: image width
        :param input_height: image height
        :param input_channels: image channels
        :param batch_size: batch size
        """

        self.img_rows = input_height
        self.img_cols = input_width
        self.channels = input_channels
        self.glob_channels = input_glob_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_glob_shape = (self.img_rows, self.img_cols, self.glob_channels)
        self.img_glob_shape_merged = (self.img_rows, self.img_cols, self.glob_channels + 1)

        self.sal_shape = (self.img_rows, self.img_cols, 1)
        self.sal_shape_merged = (self.img_rows, self.img_cols, 2)

        # self.inputWidth = input_width
        # self.inputHeight = input_height
        # self.inputChannels = input_channels

        self.G_lr = None # Generator
        self.D_lr = None # Discriminator
        self.momentum = None

        self.net = None
        self.discriminator = None
        self.batch_size = batch_size

        self.D_trainFunction = None
        self.G_trainFunction = None
        self.predictFunction = None