# BinBot

## Train the model

### Dependencies

In order to train the model, the following packages and modules are required: `Lightning`, `NumPy`, `PyTorch`, `Pandas`, `Pillow`, `Seaborn`, `Torchvision`, `Torchmetrics` and `Weights & Biases`.

To install of the dependencies in Windows 10, run the following commands in a terminal:

    python -m pip install lightning
    pip install numpy
    pip install pandas
    pip install pillow
    pip install seaborn
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install torchmetrics
    pip install wandb


**Note**: this code was tested in `python 3.11.9`.

### Dataset

The data to train the model must be in a directory named `datase\t`. Inside of this directory there must be another labeled `trash_images\` which contains `cardboard\`, `glass\` and `metal\` directories, each one with the corresponding images.

To download the dataset used [click here](https://drive.google.com/file/d/1qTWqHlfCpJmB4t92r3oxBf1KNOfoN_jM/view?usp=sharing).

### Execution

To train the model, execute the following command in a terminal:

    python training_convnext.py

## Run the BinBot

### Dependencies of the BinBot

In order to execute the BinBot, `Pillow` and `Tkinter` are required.

To install them, run in a terminal the following commands:

    pip install pillow
    pip install tk

**Note**: all the previous dependencies described in [Dependencies](###Dependencies) and [Dataset](###Dataset) are also requiered for this execution.

### Checkpoint

The BinBot required a checkpoint of the trained model in the directory `checkpoints\convnext\`. 

To download a checkpoint [click here](https://drive.google.com/file/d/15sPQ42-ZwBAwwg8D9xD34Gcwf1kXZSTr/view?usp=sharing).

### Photos

This interface also requires a `trashcans` directory with the corresponding photos of `Closed.gif`, `Cardboard.gif`, `Glass.gif` and `Metal.gif`.

### Execution

To execute the BinBot, run the following command in a terminal:

    python visualizer.py

**Note**: in order to be able to use the interface, a *webcam* is required.
