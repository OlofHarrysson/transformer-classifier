import torch
import torch.nn as nn


class Validator():
  def __init__(self, config, loss_fn):
    self.config = config
    self.loss_fn = loss_fn

  def validate(self, model, val_loader, step):
    losses = []
    accuracies = []
    model.eval()

    for batch_i, data in enumerate(val_loader, 1):
      if batch_i >= self.config.max_val_batches:
        break

      inputs, labels = data
      outputs = model.predict(inputs)

      # loss = self.loss_fn(outputs, labels)
      # losses.append(loss.item())

      _, preds = torch.max(outputs.data, 1)
      accuracy = torch.tensor(preds.cpu() == labels, dtype=float).tolist()

      accuracies.extend(accuracy)

    model.train()

    return sum(accuracies) / len(accuracies)
