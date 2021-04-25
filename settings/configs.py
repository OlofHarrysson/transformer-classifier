import anyfig
import pyjokes
import random
from datetime import datetime


# ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
@anyfig.config_class
class MiscConfig():
  def __init__(self):
    # Creates directory. Saves config & git info
    self.save_experiment: bool = False

    # An optional comment to differentiate this run from others
    self.save_comment: str = pyjokes.get_joke()

    # Start time to keep track of when the experiment was run
    self.start_time: str = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # Seed for reproducability
    self.seed: int = random.randint(0, 2**31)

    # Decides if logger should be active
    self.log_data: bool = True


# ~~~~~~~~~~~~~~ Training Parameters ~~~~~~~~~~~~~~
@anyfig.config_class
class TrainingConfig():
  def __init__(self):
    # Use GPU. Set to False to only use CPU
    # self.use_gpu: bool = True

    # Int or None
    self.gpu_idx: int = 1

    # Number of threads to use in data loading
    self.num_workers: int = 0

    # Number of update steps to train
    self.optim_steps: int = 3125

    # Number of optimization steps between validation
    self.validation_freq: int = 200

    # Number of optimization steps between validation
    self.max_val_batches: int = 50

    # Start and end learning rate for the scheduler
    self.start_lr: float = 1e-5
    self.end_lr: float = 1e-5

    # Batch size going into the network
    self.batch_size: int = 64

    # Size for image that is fed into the network
    self.input_size = 96
    # self.input_size = 32

    # Use a pretrained network
    self.pretrained: bool = False

    self.use_transformer = True
    # self.use_transformer = False
    self.n_heads = 1
    self.n_layers = 8

    self.gradient_clip = 1

    # Misc configs
    self.misc = MiscConfig()


@anyfig.config_class
class TrainLaptop(TrainingConfig):
  def __init__(self):
    super().__init__()
    ''' Change default parameters here. Like this
    self.seed = 666            ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''
    self.gpu_idx: int = None


@anyfig.config_class
class Colab(TrainingConfig):
  def __init__(self):
    super().__init__()

    self.misc.log_data: bool = True
    self.num_workers: int = 16
