import torch
import os
from model.unet_model import UNet
from loss import iou
from dataloader import get_train_data_loader,get_val_data_loader
from tqdm import tqdm
import wandb
from PIL import Image
import torchvision.transforms as T


transform = T.ToPILImage()
wandb.init(project="unet-copy")
checkpoint_dir="checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
train_data_root = "train_data"
val_data_root = "val_data"
batchsize = 2
learning_rate = 1e-5
model = UNet(3,1)
try:
    checkpoint = torch.load(checkpoint_dir+"UNet_7.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("checkpoint loaded")
except Exception:
    pass
model.cuda()
params = model.parameters()
optimizer = torch.optim.SGD(params=params,lr=learning_rate)
lossfunction = iou
train_loader = get_train_data_loader(train_data_root, batchsize=batchsize, shuffle = True)
val_loader = get_val_data_loader(val_data_root, batchsize=batchsize)



def train(train_loader, model, optimizer, epoch):
    for i, pack in enumerate(tqdm(train_loader)):
        images, masks = pack
        images = images.to('cuda')
        masks = masks.to('cuda')
        predicted_mask = model(images)
        loss = lossfunction(predicted_mask,masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }
    wandb.log({"train_loss":float(loss.item())},commit=False)
    torch.save(checkpoint, os.path.join(checkpoint_dir,f'UNet_{epoch}.pth'))

@torch.no_grad()
def val(val_loader,model):
    model.eval()
    loss = 0
    noofbatches = len(val_loader)
    for i, pack in enumerate(tqdm(val_loader)):
        images, masks = pack
        images = images.to('cuda')
        masks = masks.to('cuda')
        predicted_mask = model(images)
        loss += lossfunction(predicted_mask,masks)
    wandb.log({"val_loss":float(loss/noofbatches)},commit = False)
    wandb.log({"Image":[wandb.Image(transform(images[0]),caption="image"),\
                wandb.Image(transform(masks[0]),caption="mask"),\
                wandb.Image(transform(predicted_mask[0]),caption="prediction")]})
    print("\nvalidation_loss",loss/noofbatches,"\n")

startepoch = 8
endepoch = 11
for epoch in range(startepoch,endepoch+1):
    wandb.log({"epoch":epoch},commit=False)
    train(train_loader=train_loader,model=model,optimizer=optimizer,epoch=epoch)
    val(val_loader=val_loader,model=model)
