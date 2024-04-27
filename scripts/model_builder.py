import torch
from torch import nn 

class TinyVGG(nn.Module):
  """the CNN model

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=10, kernel_size=3, stride=1, padding=0),  
        nn.ReLU(),
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(10, 10, kernel_size=3, padding=0),
        nn.ReLU(),
        nn.Conv2d(10, 10, kernel_size=3, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=10*13*13,out_features=output_shape)
        )

  def forward(self, x: torch.Tensor):
    return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
