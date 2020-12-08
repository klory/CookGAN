# Python 3.6, PyTorch 0.4
import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import argparse
import copy
from tqdm import tqdm
from dataset_upmc import Dataset
import pdb
import wandb

import sys
sys.path.append('../../')
from common import param_counter, root
from utils_upmc import gen_filelist


# arguments
parser = argparse.ArgumentParser(description='Resnet50 UMPC-Food-101 Classifier')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--data_dir', default=f'{root}/retrieval_model/pretrain_upmc/UPMC-Food-101/')
args = parser.parse_args()
print(args)

batch_size = args.batch_size
epochs = args.epochs
data_dir = args.data_dir

# load data
parts = ('train', 'test')
for part in parts:
    gen_filelist(data_dir, part)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
    parts[0]: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    parts[1]: transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

datasets = {
    x: Dataset(
        root=os.path.join(data_dir, 'images'), 
        flist=os.path.join(data_dir, x+".txt"), 
        transform=data_transforms[x]) 
    for x in parts}

# datasets = {x: Subset(datasets[x], range(200)) for x in parts}

dataloaders = {
    x: torch.utils.data.DataLoader(
        datasets[x], 
        batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=False)
    for x in parts}

dataset_sizes = {x: len(datasets[x]) for x in parts}
dataloader_sizes = {x: len(dataloaders[x]) for x in parts}
print('datasets', dataset_sizes)
print('dataloaders', dataloader_sizes)

#  load model
model = models.resnet50(pretrained=True)
num_feat = model.fc.in_features
model.fc = nn.Linear(num_feat, 101)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
if device == 'cuda':
    model = nn.DataParallel(model)
    model_to_save = model.module
else:
    model_to_save = model

print('# parameters:', param_counter(model.parameters()))

wandb.init(project="cookgan_pretrain_upmc")
wandb.config.update(args)

# train
for epoch in range(epochs):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    running_loss = 0.0
    running_correct = 0.0
    for part in parts:
        if part == 'train':
            model.train()
        else:
            model.eval()

        pbar = tqdm(dataloaders[part])
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(part == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                pbar.set_description(f'loss={loss:.4f}')
                if part == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.shape[0]
            running_correct += torch.sum(preds == labels)

        loss_epoch = running_loss / dataset_sizes[part]
        acc_epoch = running_correct.double() / dataset_sizes[part]
        log = {
            'epoch': epoch,
            f'loss_{part}': loss_epoch,
            f'accuracy_{part}': acc_epoch,
            'lr': optimizer.param_groups[0]['lr']
        }
    
    scheduler.step(loss_epoch)
    if epoch % 5 == 0:
        print('save checkpoint...')
        torch.save(model_to_save.state_dict(), f'{wandb.run.dir}/{epoch:>06d}.ckpt')