import configuration as config
import torch 

from models.convnext import ConvNext
from models.image_classification_lightning_module import ImageClassificationLightningModule
from torch import nn
from helper_functions import count_classes
from torchmetrics import MetricCollection


def load_checkpoint():
  # Configure environment
  torch.set_float32_matmul_precision("high")
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  class_count = count_classes(config.ROOT_DIR)

  # Unused parameters
  metrics = MetricCollection({})
  vector_metrics = MetricCollection({})

  # Create model
  convnext = ConvNext(num_classes=class_count, device=device)
  model = ImageClassificationLightningModule.load_from_checkpoint(
    checkpoint_path="checkpoints/convnext/convnext_.ckpt",
    model = convnext,
    loss_fn=nn.CrossEntropyLoss(),
    metrics=metrics,
    vectorized_metrics=vector_metrics,
    lr=config.LR,    
    scheduler_max_it=config.SCHEDULER_MAX_IT,
  ).model

  # Do not train the model
  model.eval()

  return model

def main():
  load_checkpoint()

if __name__ == "__main__":
  main()