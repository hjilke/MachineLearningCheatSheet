from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import re
import torch
import os



##########################################################
# Specify path to image files
##########################################################
root = os.path.dirname(__name__)
DATA_DIR = "data"
IMAGE_FOLDER_TEST = os.path.join(root, DATA_DIR, "test")
IMAGE_FOLDER_TRAIN = os.path.join(root, DATA_DIR, "train")


class ImageDataSet(Dataset):
    """
    Custom PyTorch Imageloader for Cats vs. Dogs data set see link  
    for details https://www.kaggle.com/c/dogs-vs-cats.  
    """
    def __init__(self, root_dir, transform, extensions=[".png", ".jpg"]):
        self.root_dir = root_dir
        self.transform = transform
        self.total_imgs = os.listdir(root_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.total_imgs[idx])
        sample = Image.open(img_name).convert("RGB")
        if self.transform:
            sample = self.transform(sample)

        label = 0 if "cat" in img_name else 1
        return sample, label



def plot_helper(image, ax=None, title=None, normalize=True):
    
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ]) 

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

train_data = ImageDataSet(IMAGE_FOLDER_TRAIN, transform=train_transforms)
test_data = ImageDataSet(IMAGE_FOLDER_TEST, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)


data_iter = iter(trainloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), nrows=1, ncols=4)
for i, ax in enumerate(axes):
    plot_helper(images[i], ax=ax, normalize=False)

plt.show()