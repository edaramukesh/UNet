import torch
import os
import cv2
import torchvision.transforms as T
from model.unet_model import UNet
from glob import glob
from PIL import Image
from tqdm import tqdm
from loss import iou

img_transform = T.Compose([
                T.Resize((500,500)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


@torch.no_grad()
def trace_model():
    model = UNet(3,1)
    checkpoint = torch.load("checkpoints/UNet_7.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    images = glob(f'trace_data/*.jpg')
    inputs = torch.tensor([])
    for address in images:
        img = img_transform(Image.open(address)).unsqueeze(0)
        inputs = torch.cat((inputs, img), 0)
    traced_model =torch.jit.trace(model,inputs)
    torch.jit.save(traced_model,"traced_models/traced_model_nogradtest.pt")

@torch.no_grad()
def Inference_using_traced_model():
    traced_model = torch.jit.load('traced_models/traced_model_7_3img.pt')
    traced_model.cuda()
    input_images_names = glob("test_data/*.jpg")
    print("number of input images =",len(input_images_names))
    for idx, image_path in enumerate(tqdm(input_images_names)):
        image = Image.open(image_path).convert('RGB')
        name = image_path.split('/')[-1].split('.')[0]
        # image = T.functional.pil_to_tensor(image)
        image = img_transform(image)
        # image = image.float()
        image = image.unsqueeze(0)
        image = image.to('cuda')
        mask = traced_model(image)
        mask = mask.mul(255)
        mask = mask.to(torch.uint8)
        mask = mask.data.cpu().numpy().squeeze()
        cv2.imwrite(f'trace_output/{name}.jpg', mask)

@torch.no_grad()
def traceLoss():
    input_images_names = glob("train_data/images/*.jpg")
    traced_model = torch.jit.load('traced_models/traced_model_7_3img.pt')
    traced_model.cuda()
    model = UNet(3,1)
    checkpoint = torch.load("checkpoints/UNet_7.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = 0
    for idx, image_path in enumerate(tqdm(input_images_names[:5])):
        image = Image.open(image_path).convert('RGB')
        image = img_transform(image)
        image = image.unsqueeze(0)
        maskmodel = model(image)
        maskmodel = maskmodel.mul(255)
        maskmodel = maskmodel.to(torch.uint8)

        image = image.to('cuda')
        # print(image.is_cuda)
        mask = traced_model(image)
        mask = mask.mul(255)
        mask = mask.to(torch.uint8)
        # mask = mask.data.cpu().numpy().squeeze()
        maskmodel = maskmodel.to('cuda')
        loss += iou(mask,maskmodel)
        # print(loss/idx+1)
        
    print("\nTraced model loss =",loss/5)

@torch.no_grad()
def scriptmodel():
    model = UNet(3,1)
    checkpoint = torch.load("checkpoints/UNet_7.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    scripted_model =torch.jit.script(model)
    torch.jit.save(scripted_model,"traced_models/scripted_model_7.pt")

@torch.no_grad()
def Inference_using_scripted_model():
    scripted_model = torch.jit.load('traced_models/scripted_model_7.pt')
    scripted_model.cuda()
    input_images_names = glob("test_data/*.jpg")
    print("number of input images =",len(input_images_names))
    for idx, image_path in enumerate(tqdm(input_images_names)):
        image = Image.open(image_path).convert('RGB')
        name = image_path.split('/')[-1].split('.')[0]
        # image = T.functional.pil_to_tensor(image)
        image = img_transform(image)
        # image = image.float()
        image = image.unsqueeze(0)
        image = image.to('cuda')
        mask = scripted_model(image)
        mask = mask.mul(255)
        mask = mask.to(torch.uint8)
        mask = mask.data.cpu().numpy().squeeze()
        cv2.imwrite(f'scripted_output/{name}.jpg', mask)
    
@torch.no_grad()
def ScriptLoss():
    input_images_names = glob("train_data/images/*.jpg")
    scripted_model = torch.jit.load('traced_models/scripted_model_7.pt')
    scripted_model.cuda()
    model = UNet(3,1)
    checkpoint = torch.load("checkpoints/UNet_7.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = 0
    for idx, image_path in enumerate(tqdm(input_images_names[:5])):
        image = Image.open(image_path).convert('RGB')
        image = img_transform(image)
        image = image.unsqueeze(0)
        maskmodel = model(image)
        maskmodel = maskmodel.mul(255)
        maskmodel = maskmodel.to(torch.uint8)

        image = image.to('cuda')
        # print(image.is_cuda)
        mask = scripted_model(image)
        mask = mask.mul(255)
        mask = mask.to(torch.uint8)
        # mask = mask.data.cpu().numpy().squeeze()
        maskmodel = maskmodel.to('cuda')
        loss += iou(mask,maskmodel)
        # print(loss/idx+1)
        
    print("\nScripted model loss =",loss/5)


if __name__ == "__main__":
    # trace_model()
    # Inference_using_traced_model()
    # traceLoss()
    # scriptmodel()
    # Inference_using_scripted_model()
    # ScriptLoss()
    pass