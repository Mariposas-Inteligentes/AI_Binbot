# Imports a module
from torch import nn
import torchvision

import configuration as conf

# Import ConvNext (A ConvNet for the 2020s) with it's weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# Inherits from nn.Module
class ConvNext(nn.Module):
    # Device sets where to run (CPU, GPU, TPU...etc)
    def __init__(self, num_classes, device):
        super().__init__()
        # Create the module, set the wieghts and orient it to the device we want
        self.convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).to(
            device=device
        )

        # Parameters refer to our tensors
        for parameter in self.convnext.parameters():
            # Set as pretrained with no change if set to false
            parameter.requires_grad = False

        # Set as classifier for our specific task (add a perceptron)
        # classifier [2] lets us change the classification layer
        # in_features -> neuron
        # bias -> bias
        # out_features -> number of classes
        # Only this will be trained if the other line is set as false (parameter.requires_grad = True)
        # Otherwise, the neural network will fine tune
        self.convnext.classifier[2] = nn.Linear(
            in_features=768, out_features=num_classes, bias=False
        )

    # Operation to be done
    def forward(self, x):
        return self.convnext(x)

    @staticmethod
    def get_transformations() -> tuple[torchvision.transforms.Compose]:
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(conf.RESIZE),
                torchvision.transforms.CenterCrop(conf.CROP),
                torchvision.transforms.ToTensor(),
                # Adds extra transformations to create more data -> Depends on data it might make sense
                # Lets us do data augmentation, we can add more RandomOperations
                torchvision.transforms.RandomRotation(conf.ROTATION),
                torchvision.transforms.Normalize(conf.MEAN, conf.STD),
            ]
        )

        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(conf.RESIZE),
                torchvision.transforms.CenterCrop(conf.CROP),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(conf.MEAN, conf.STD),
            ]
        )
        return train_transform, test_transform
