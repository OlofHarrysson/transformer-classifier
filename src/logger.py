import visdom
import functools
from anyfig import cfg
from .utils import plotly_plots as plts
import plotly.graph_objects as go
import torch
import numpy as np


def clear_old_data(vis):
  [vis.close(env=env) for env in vis.get_env_list()]  # Kills wind
  # [vis.delete_env(env) for env in vis.get_env_list()] # Kills envs


def log_if_active(func):
  ''' Decorator which only calls logging function if logger is active '''
  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    if cfg().misc.log_data:
      return func(self, *args, **kwargs)

  return wrapper


class Logger():
  def __init__(self):
    if cfg().misc.log_data:
      try:
        self.vis = visdom.Visdom()
        clear_old_data(self.vis)
      except Exception as e:
        err_msg = "Couldn't connect to Visdom. Make sure to have a Visdom server running or turn of logging in the config"
        raise ConnectionError(err_msg) from e

  @log_if_active
  def log_image(self, image):
    self.vis.image(image)

  @log_if_active
  def log_accuracy(self, accuracy, step, name='train'):
    title = f'{name} Accuracy'.title()
    plot = plts.accuracy_plot(self.vis.line, title)
    plot(X=[step], Y=[accuracy])

  @log_if_active
  def log_parameters(self, text):
    self.vis.text(text.replace('\n', '<br>'))

  @log_if_active
  def log_gradients(self, model, layer_name):
    layers = []
    ave_grads = []
    max_grads = []

    for name, param in model.named_parameters():
      if param.grad is None or 'bias' in name or layer_name not in name:
        continue

      layers.append(name)
      ave_grads.append(param.grad.abs().mean().item())
      max_grads.append(param.grad.abs().max().item())

    Y = np.array([ave_grads, max_grads]).T
    title = f'gradients {layer_name}'.title()
    opts = dict(title=title)
    self.vis.line(Y=Y, win=title, opts=opts)
