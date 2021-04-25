import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from . import resnet
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def get_model(config):
  model = MyModel(config)
  model = model.to(model.device)
  return model


class MyModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cpu' if config.gpu_idx is None else torch.device(
      'cuda', config.gpu_idx)

    self.backbone = resnet.resnet50(pretrained=config.pretrained)

    self.use_transformer = config.use_transformer

    # Transformer
    n_features = 1024 * 2
    n_hid = n_features
    norm = nn.LayerNorm(n_features)
    encoder_layers = TransformerEncoderLayer(n_features, config.n_heads, n_hid)
    self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                  config.n_layers, norm)

    # Finish
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    n_classes = 10
    self.fc = nn.Linear(n_features, n_classes)

  def forward(self, inputs):
    inputs = inputs.to(self.device)
    x = self.backbone(inputs)

    if self.use_transformer:
      height = x.shape[-1]
      x = x.reshape(x.shape[0], x.shape[1], -1)
      x = x.transpose(1, -1)

      # Transformer
      x = self.transformer_encoder(x)
      x = x.transpose(1, -1)
      x = x.reshape(x.shape[0], x.shape[1], height, height)

    x = self.avgpool(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc(x)

    return x

  def predict(self, inputs):
    with torch.no_grad():
      return self(inputs)

  def save(self, path):
    path = Path(path)
    err_msg = f"Expected path that ends with '.pt' or '.pth' but was '{path}'"
    assert path.suffix in ['.pt', '.pth'], err_msg
    path.parent.mkdir(exist_ok=True)
    print("Saving Weights @ " + str(path))
    torch.save(self.state_dict(), path)

  def load(self, path):
    print('Loading weights from {}'.format(path))
    weights = torch.load(path, map_location='cpu')
    self.load_state_dict(weights, strict=False)
    self.to(self.device)
