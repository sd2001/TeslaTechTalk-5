import torch
import torch.nn as nn
import torch.optim as optmin
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image 
import requests
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))],
)
model = models.resnet50(pretrained=True)
model.eval()
model.to(device)
in_channels = model.fc.in_features
out_channels = model.fc.out_features
in_channels,out_channels
model.fc=nn.Sequential(nn.Linear(in_channels, out_channels),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(out_channels,10))
                         
model = model.to(device)

path = "./src/cifar10_complete50.pth"

model.load_state_dict(torch.load(path, map_location="cpu"))
model = model.to(device)
categories = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

def image_custom(img):
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.to(device)
    input1 = Variable(image_tensor)   
    input1 = input1.to(device)
    output = model(input1)
    _, index = output.max(1)
    return index

def predict(url):
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)  
    img = img.resize((32,32))
    img = np.array(img)
    to_pil = transforms.ToPILImage()
    img = to_pil(img)
    index = image_custom(img).item()
    return categories[index]

# print(predict("https://i.insider.com/5cbf50dfd1a2f8074406a8b2?width=700"))