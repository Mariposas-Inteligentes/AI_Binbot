import configuration as config
import torch 

from models.convnext import ConvNext
from models.image_classification_lightning_module import ImageClassificationLightningModule
from torch import nn
from helper_functions import count_classes
from torchmetrics import MetricCollection
from torchvision import transforms
from PIL import Image


class TrainedConvNext():
  def __init__(self):
    # Configure the environment
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.class_count = count_classes(config.ROOT_DIR)
    torch.set_float32_matmul_precision("high")

    # Create the model
    metrics = MetricCollection({})
    vector_metrics = MetricCollection({})

    # Create model
    self.convnext = ConvNext(num_classes=self.class_count, device=self.device)
    self.model = ImageClassificationLightningModule.load_from_checkpoint(
      checkpoint_path="checkpoints/convnext/convnext_.ckpt",
      model = self.convnext,
      loss_fn=nn.CrossEntropyLoss(),
      metrics=metrics,
      vectorized_metrics=vector_metrics,
      lr=config.LR,    
      scheduler_max_it=config.SCHEDULER_MAX_IT,
    ).model

    # Ensure model is in evaluation mode
    self.model.eval()

    train_transform, self.test_transform = ConvNext.get_transformations()

  # Load and preprocess the image
  def load_image(self, image_path):
    image = Image.open(image_path).convert('RGB')
    return self.test_transform(image).unsqueeze(0)

  # Function to predict the class of an image
  def predict(self, image_path):
      # Load image and move it to the correct device
      image_tensor = self.load_image(image_path).to(self.device)

      # Disable gradient calculation for inference
      with torch.no_grad():
          outputs = self.model(image_tensor)
      
      # Get the predicted class (argmax of the output probabilities)
      _, predicted_class = outputs.max(1)
      
      return predicted_class.item()

def test():
  print("Creating the model...")
  trained_model = TrainedConvNext()
  print("Model created! Predicting...")
  image_path = 'dataset/trash_images/cardboard/cardboard32.jpg'
  predicted_class = trained_model.predict(image_path)
  print(f'Predicted class: {config.CLASS_NAMES[predicted_class]}')