#in this file i will create a pipeline to predict output for a given input of data , 
#return an output that is a csv
#containing (name/ID , predicted_class)

import os
import torch
import pandas as pd
from pathlib import Path
import data_setup, engine, model_builder, utils
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    
    #initilalizing the paths
    prefix=Path("../data/pizza_steak_sushi/test")
    sample_submission=pd.read_csv('../data/pizza_steak_sushi/SampleSubmission.csv')
    paths=[prefix / Path(row+".jpg") for row in sample_submission['Image_ID']]
    # Specify column names
    columns = ['Image_ID', 'Pizza','Sushi','Steak'] #this matters a lot
    output_df = pd.DataFrame(columns=columns)
    output_df['Image_ID']=[str(path)[32:-4] for path in paths] #32 to remove the useless path part , -3 to remove the extension of the image , all of this to just leave the name
    #load model after initializing it
    model_name="model1.pth"
    model_path=Path("../models") / model_name
    model=model_builder.TinyVGG(input_shape=3,output_shape=3).to(device)
    model.load_state_dict(torch.load(model_path))
    #specifying parameters
    NUM_WORKERS=1
    BATCH_SIZE=16
    #specifying which data_transform we gonna use
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    submission_dataloader=data_setup.create_submit_dataloader(
        path_strings=paths,
        transform=data_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    model.eval()
    y=[]
    for batch,X in enumerate(submission_dataloader):
        with torch.inference_mode():
            for elm in torch.argmax(torch.softmax(model(X),dim=1),dim=1).tolist():
                y.append(elm)

    def fill(output_df,model_prediction):
        for idx, prediction in enumerate(model_prediction):
            output_df.loc[idx, 'Pizza'] = 1 if prediction == 0 else 0
            output_df.loc[idx, 'Sushi'] = 1 if prediction == 1 else 0
            output_df.loc[idx, 'Steak'] = 1 if prediction == 2 else 0

    fill(output_df, y)

    output_df.to_csv('predictions.csv', index=False)