from bayesianSimpleViT import BayesianSimpleViT
from bayesianViT import BayesianViT
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm, trange
import sklearn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import torch.nn as nn
import torchbnn as bnn

from torch.utils.data.dataloader import default_collate


from bayesianLinformer import BayesianLinformer
from linformer import Linformer

efficient_transformer = BayesianLinformer(
    dim=256,
    seq_len=16+1,  # 4x4 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64,              
    dropout=0.1
)

model = BayesianViT(
    dim=256,
    image_size=28,
    patch_size=7,
    num_classes=10,
    transformer=efficient_transformer,
    channels=1,
)

model.load_state_dict(torch.load('bigBayesianViTmodel.pth'))
model = model.cuda() if torch.cuda.is_available() else model

images = torch.rand(10, 1, 28, 28).cuda()
out = model(images)
print(out)

from torchsummary import summary
summary(model, input_size=(1, 28, 28))




# Load image data from Cats vs Dogs dataset
# define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomResizedCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# define the path to the root folder
#root = 'CatsvsDogs/train'

# create the ImageFolder dataset
#dataset = datasets.ImageFolder(root=root, transform=transform)
batch_size = 1024

train_dataset = datasets.MNIST(
    root='./dataset/minst/',
    train=True,
    download=False,
    transform=transform
)


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))


test_dataset = datasets.MNIST(
    root='./dataset/minst/',
    train=False,
    download=False,
    transform=transform
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))


# calculate the size of the training set and testing set
#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size

# randomly split the dataset into training and testing sets
#train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# create data loaders for the training and testing sets
#batch_size = 1024
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))

# print the classes
#print(dataset.classes)

# print the label per class
#print(dataset.class_to_idx)

# print the number of images in the dataset
#print(len(dataset))            

model.train()
training_iter = 400
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
#scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
ce_loss = nn.CrossEntropyLoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 1

for i in range(training_iter):

    losses = []
    correctAcc, totalAcc = 0, 0

    for data in tqdm(train_loader, desc=f"Epoch {i + 1} in training", leave=False):


        x_batch, y_batch = data

        # Forward pass
        out = model(x_batch)

        # Calculate loss
        ce = ce_loss(out, y_batch)
        kl = kl_loss(model)
        loss_value = ce + kl_weight*kl

        # Backward pass
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())

        correctAcc += torch.sum(torch.argmax(out, dim=1) == y_batch).detach().cpu().item()
        totalAcc += len(x_batch)


    results = []
    true_values = []
    val_losses = []

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for test_mini_epoch, val_data in enumerate(test_loader):


            x_batch, y_batch = val_data


            # Forward pass
            out = model(x_batch)
            results.append(out.cpu().numpy())
            true_values.append(y_batch.cpu().numpy())

            # Calculate loss
            ce = ce_loss(out, y_batch)
            kl = kl_loss(model)
            loss_value = ce + kl_weight*kl

            val_losses.append(loss_value.item())

            correct += torch.sum(torch.argmax(out, dim=1) == y_batch).detach().cpu().item()
            total += len(x_batch)
                

    model.train()


    val_accuracy = correct / total 
    train_accuracy = correctAcc / totalAcc


    print('Iter : %d / %d --- Training Loss : %.3f --- Training Accuracy : %.3f --- Validation Loss : %.3f --- Validation Accuracy : %.3f' % (i+1, training_iter, np.mean(losses), train_accuracy, np.mean(val_losses), val_accuracy))


torch.save(model.state_dict(), 'bigBayesianViTmodel.pth')