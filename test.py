import torch
import os
import cv2
import torchvision.transforms as T
from PIL import Image
from glob import glob
from model.unet_model import UNet
from tqdm import tqdm

checkpoint = torch.load("checkpoints/UNet_7.pth")
input_images_names = glob("test_data/*.jpg")
print("number of input images =",len(input_images_names))

output_dir = 'test_output'

if not os.path.exists("test_output"):
    os.mkdir(output_dir)

model = UNet(3,1)
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()
model.eval()
# print(next(model.parameters()).is_cuda)

img_transform = T.Compose([
                T.Resize((500,500)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

for idx, image_path in enumerate(tqdm(input_images_names)):
    image = Image.open(image_path).convert('RGB')
    name = image_path.split('/')[-1].split('.')[0]
    # image = T.functional.pil_to_tensor(image)
    image = img_transform(image)
    # image = image.float()
    image = image.unsqueeze(0)
    image = image.to('cuda')
    # print(image.is_cuda)
    with torch.inference_mode():
        mask = model(image)
    mask = mask.mul(255)
    mask = mask.to(torch.uint8)
    mask = mask.data.cpu().numpy().squeeze()
    cv2.imwrite(f'test_output/{name}.jpg', mask)