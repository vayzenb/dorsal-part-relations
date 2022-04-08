import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import natsort
from PIL import Image

class LoadFrames(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        
        return tensor_image, self.total_imgs[idx]


""" transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
hab_data = LoadFrames('Frames/Figure_23_Bulge',transform_)
trainloader = torch.utils.data.DataLoader(hab_data, batch_size=len(hab_data), shuffle=False, num_workers = 4, pin_memory=True)
#dataiter = iter(trainloader)
print(hab_data) """