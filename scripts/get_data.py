#you should run this file if you you data is in .zip form and you wanna unzip it and put it inside its intended directory
import sys
import os
import requests
import zipfile
from pathlib import Path

import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:#sys.argv is an array that contains the name of the file to be run by python and the arguments passed in , so a len(sys.argv)==number of arguments passed in +1
        print("Usage: python get_data.py <directory> <dataset_name> <url>")
    else:
        directory = sys.argv[1]
        dataset_name = sys.argv[2]
        url = sys.argv[3]
        print("Directory:", directory)
        print("Dataset Name:", dataset_name)
        print("URL:", url)
        # Now you can use the URL with requests.get()



#when u run this file , all you need to change are these variables, run from terminal command
#__start__
parent_data_folder_path=directory #this needs to exist before running this file
data_folder_name=dataset_name #this will be created after running this file , or it will be detected if it's already created
link_for_data=url
#__end__

# Setup path to data folder
data_path = Path(parent_data_folder_path+"/")
image_path = data_path / data_folder_name

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)


# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get(link_for_data)
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")