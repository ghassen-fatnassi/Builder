#this file is basically a function that creates a dataloader for val and another for training and return them 
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import random
import pathlib
from PIL import Image

####################################### CUSTOM DATASET PART ##########################################33
# Make function to find classes in target directory
def find_classes(directory: str)-> tuple[list[str], dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int)-> tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

class ImageFolderCustomForTest(ImageFolderCustom):
    def __init__(self, path_strings: list[str], transform=None) -> None:
        # . Create class attributes
        # Get all image paths
        self.paths = [pathlib.Path(path_string) for path_string in path_strings] # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform

    # . Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int)-> torch.Tensor:
        "Returns one sample of data"
        img = self.load_image(index)
        # Transform if necessary
        if self.transform:
            return self.transform(img) # return data-transformed
        else:
            return img # return data

# # 1. Take in a Dataset as well as a list of class names
# def display_random_images(dataset: torch.utils.data.dataset.Dataset,
#                           classes: list[str] = None,
#                           n: int = 10,
#                           display_shape: bool = True,
#                           seed: int = None):
    
#     # 2. Adjust display if n too high
#     if n > 10:
#         n = 10
#         display_shape = False
#         print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
#     # 3. Set random seed
#     if seed:
#         random.seed(seed)

#     # 4. Get random sample indexes
#     random_samples_idx = random.sample(range(len(dataset)), k=n)

#     # 5. Setup plot
#     plt.figure(figsize=(16, 8))

#     # 6. Loop through samples and display random samples 
#     for i, targ_sample in enumerate(random_samples_idx):
#         targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

#         # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
#         targ_image_adjust = targ_image.permute(1, 2, 0)

#         # Plot adjusted samples
#         plt.subplot(1, n, i+1)
#         plt.imshow(targ_image_adjust)
#         plt.axis("off")
#         if classes:
#             title = f"class: {classes[targ_label]}"
#             if display_shape:
#                 title = title + f"\nshape: {targ_image_adjust.shape}"
#         plt.title(title)

# # Display random images from ImageFolder created Dataset
# display_random_images(train_data, 
#                       n=5, 
#                       classes=class_names,
#                       seed=None)

############################################ BASIC DATASET PART #####################################
def create_train_dataloader(train_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int):

  data_custom = ImageFolderCustom(targ_dir=train_dir, 
                                      transform=transform)

  # Get class names
  class_names = data_custom.classes

  # Turn images into data loaders
  dataloader = DataLoader(
      data_custom,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )

  return dataloader, class_names

def create_val_dataloader(val_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int):

  data_custom = ImageFolderCustom(targ_dir=val_dir, 
                                      transform=transform)

  # Get class names
  class_names = data_custom.classes

  # Turn images into data loaders
  dataloader = DataLoader(
      data_custom,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return dataloader, class_names

def create_submit_dataloader(path_strings: str, transform: transforms.Compose, batch_size: int, num_workers: int):

  data_custom = ImageFolderCustomForTest(path_strings=path_strings, 
                                      transform=transform)

  # Turn images into data loaders
  dataloader = DataLoader(
      data_custom,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return dataloader