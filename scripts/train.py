import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
import sys

if __name__ == "__main__":
  if len(sys.argv) != 5:#sys.argv is an array that contains the name of the file to be run by python and the arguments passed in , so a len(sys.argv)==number of arguments passed in +1
        print("Usage: python train.py <NUM_EPOCHS> <BATCH_SIZE> <NUM_WORKERS> <LEARNING_RATE>")
  else:
        NUM_EPOCHS = sys.argv[1]
        BATCH_SIZE = sys.argv[2]
        NUM_WORKERS = sys.argv[3]
        LEARNING_RATE = sys.argv[4]
        print("NUM_EPOCHS:", NUM_EPOCHS)
        print("BATCH_SIZE:", BATCH_SIZE)
        print("NUM_WORKERS:", NUM_WORKERS)
        print("LEARNING_RATE:", LEARNING_RATE)
        NUM_EPOCHS=int(NUM_EPOCHS)
        NUM_WORKERS=int(NUM_WORKERS)
        BATCH_SIZE=int(BATCH_SIZE)
        LEARNING_RATE=float(LEARNING_RATE)
        # Setup directories
        train_dir = "../data/pizza_steak_sushi/train"
        val_dir = "../data/pizza_steak_sushi/eval"

        # Setup target device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create transforms
        data_transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
        ])

        # Create training DataLoaders with help from data_setup.py
        train_dataloader, class_names = data_setup.create_train_dataloader(
            train_dir=train_dir,
            transform=data_transform,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        # Create evaluation DataLoaders with help from data_setup.py
        val_dataloader, class_names = data_setup.create_val_dataloader(
            val_dir=val_dir,
            transform=data_transform,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

        # Create model with help from model_builder.py
        model = model_builder.TinyVGG(
            input_shape=3,
            output_shape=len(class_names)
        ).to(device)

        # Set loss and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=LEARNING_RATE)


        # Start training with help from engine.py
        engine.train(model=model,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS,
                        device=device)

        # Save the model with help from utils.py
        utils.save_model(model=model,
                        target_dir="../models",
                        model_name="model1.pth")