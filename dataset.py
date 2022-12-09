import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from glob import glob



class ExteriorData(Dataset):

    def __init__(self, root,transform = True, phase = 'train'):
        self.root = root
        self.transform = transform
        self.phase = phase
        self.imgs = glob(f'{self.root}/images/*.jpg') + glob(f'{self.root}/images/*.JPG') + \
                    glob(f'{self.root}/images/*.png') + glob(f'{self.root}/images/*.PNG')
        self.masks = glob(f'{self.root}/mask/*.jpg') + glob(f'{self.root}/mask/*.JPG') + \
                    glob(f'{self.root}/mask/*.png') + glob(f'{self.root}/mask/*.PNG') 
        noimgs = len(self.imgs)
        nomasks = len(self.masks)
        
        print(f"num of images given = {noimgs} , num of masks given = {nomasks}")
        
        self.new_imgs, self.new_masks = [], []
        self.imgs.sort()
        self.masks.sort()
        if nomasks<=noimgs:
            i=0
            j=0
            while i<nomasks and j<noimgs:
                if self.masks[i].replace("mask","images")[:-4] == self.imgs[j][:-4]:
                    self.new_masks.append(self.masks[i])
                    self.new_imgs.append(self.imgs[j])
                    i+=1
                    j+=1
                elif self.masks[i].replace("mask","images")[:-4] < self.imgs[j][:-4]:
                    i+=1
                else:
                    j+=1
        else:
            i=0
            j=0
            while i<noimgs and j<nomasks:
                if self.imgs[i].replace("images","mask")[:-4] == self.masks[j][:-4]:
                    self.new_imgs.append(self.imgs[i])
                    self.new_masks.append(self.masks[j])
                    i+=1
                    j+=1
                elif self.imgs[i].replace("images","mask")[:-4] < self.masks[j][:-4]:
                    i+=1
                else:
                    j+=1
        
        print(f"num of images= {len(self.new_imgs)} , num of masks = {len(self.new_masks)}")
        
        if transform:
            self.img_transform = T.Compose([
                T.Resize((500,500)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.mask_transform = T.Compose([
                T.Resize((500,500)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.new_imgs)

    def __getitem__(self, idx):
        if self.transform:
            img = Image.open(self.new_imgs[idx]).convert('RGB')
            mask =  Image.open(self.new_masks[idx]).convert('L')
            img = self.img_transform(img)
            mask = self.mask_transform(mask)
        return img,mask
    