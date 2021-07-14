import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from PIL import Image
  
import glob
import os

# dataset
class myDataset(Dataset):
    def __init__(self,root,transform):
        self.root = root                                            # input path
        self.transform = transform                                  
        self.file = sorted(glob.glob(os.path.join(root,'*.jpg')))   # read data's file name 

    def __getitem__(self,index):
        img = Image.open(self.file[index])
        item = self.transform(img)
        return item

    def __len__(self):
        return len(self.file)

# dataloder
root = "/home/user1/project/project/Dataset/test/"
transform = transforms.Compose([transforms.Resize((80, 80)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

mydataloader = DataLoader(dataset=myDataset(root,transform),
                          batch_size=3,
                          shuffle=False)

to_image = transforms.ToPILImage()                                  # tensor to image

# show the images
for image in mydataloader:
    img = to_image(image[0])
    img.show()
