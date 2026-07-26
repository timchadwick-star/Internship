import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
import torchmetrics

from torchvision.models import resnet18, ResNet18_Weights

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

torch.manual_seed(42)
#create transform so resnet18 can use the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

#setup data
train_data = datasets.MNIST(
    root="data",
    train=True, 
    download=True, 
    transform=transform, 
    target_transform=None
)

test_data = datasets.MNIST(
    root="data",
    train=False, 
    download=True, 
    transform=transform, 
    target_transform=None
)

from torch.utils.data import DataLoader

# Setup the batch size hyperparameter and number of epochs
BATCH_SIZE = 32
epochs = 3

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Select 100 random indices from the test dataset
random_indices = torch.randperm(len(test_data))[:100]

# Create a subset containing those 100 images
test_subset = torch.utils.data.Subset(
    test_data,
    random_indices
)

# Create a DataLoader for the subset
test_subset_dataloader = DataLoader(
    test_subset,
    batch_size=32,
    shuffle=False
)

# Load pretrained ResNet18
weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer with a 10-class classifier
model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=10
)

model.to("cpu")

# Add loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer only updates the new final layer
optimizer = torch.optim.SGD(
    model.fc.parameters(),
    lr=0.1
)

accuracy = torchmetrics.Accuracy(
    task="multiclass",
    num_classes=10
)

#train

results = []

for epoch in range(epochs):

    # Temporary storage for the last 375 batches
    running_loss = 0
    running_accuracy = 0
    batch_count = 0

    for batch, (X, y) in enumerate(train_dataloader):
        model.train()
        #Forward pass
        y_pred = model(X)

        #Calculate loss (per batch)
        loss = loss_fn(y_pred, y)

        # Calculate accuracy for current batch
        batch_accuracy = accuracy(y_pred, y)

        running_loss += loss.item()
        running_accuracy += batch_accuracy.item()
        batch_count +=1

        #Optimizer zero grad
        optimizer.zero_grad()

        #Loss backward
        loss.backward()

        #Optimizer step
        optimizer.step()

        #print how many samples have been looked at
        if batch % 375 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

            average_train_loss = running_loss / batch_count
            average_train_accuracy = running_accuracy / batch_count

            #test

            model.eval()
            test_loss = 0
            test_accuracy = 0

            with torch.inference_mode():

                for X_test, y_test in test_subset_dataloader:

                    # Forward pass
                    test_pred = model(X_test)

                    # Calculate test loss
                    loss = loss_fn(test_pred, y_test)

                    # Calculate test accuracy
                    acc = accuracy(test_pred, y_test)

                    # Add to totals
                    test_loss += loss.item()
                    test_accuracy += acc.item()

            # Average over all test batches
            average_test_loss = test_loss / len(test_subset_dataloader)
            average_test_accuracy = test_accuracy / len(test_subset_dataloader)

            total_batch = epoch * len(train_dataloader) + batch

            #Record results in a table
            results.append({
                "Epoch": epoch,
                "Batch": total_batch,
                "Train Loss": average_train_loss,
                "Train Accuracy": average_train_accuracy,
                "Test Loss": average_test_loss,
                "Test Accuracy": average_test_accuracy
            })

            #reset running totals
            running_loss = 0
            running_accuracy = 0
            batch_count = 0


#store results in dataframe
df = pd.DataFrame(results).round(3)
print(df)
df.to_csv("resnet_results.csv")

# -------------------------
# LOSS CURVE
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["Batch"],
    df["Train Loss"],
    label="Training Loss"
)

plt.plot(
    df["Batch"],
    df["Test Loss"],
    label="Test Loss"
)

plt.xlabel("Training Batch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid()

plt.savefig("resnet_loss_curve.png", dpi=300)

plt.show()


# -------------------------
# ACCURACY CURVE
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["Batch"],
    df["Train Accuracy"],
    label="Training Accuracy"
)

plt.plot(
    df["Batch"],
    df["Test Accuracy"],
    label="Test Accuracy"
)

plt.xlabel("Training Batch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.grid()

plt.savefig("resnet_accuracy_curve.png", dpi=300)

plt.show()