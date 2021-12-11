import torch
from torchvision import models
import cv2

vgg = models.vgg19(pretrained=True)
vgg = vgg.features
#print(vgg)

for parameters in vgg.parameters():
    parameters.requires_grad_(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
vgg.to(device)

from PIL import Image
from torchvision import transforms as T

def preprocess(img_path,max_size=620):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    img_transforms = T.Compose([T.Resize(size),T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    image = img_transforms(image)
    image = image.unsqueeze(0)
    return image

content_img_processed = preprocess('/home/anubhav/Projects/NST/data/content/content10.jpg').to(device)
style_img_processed = preprocess('/home/anubhav/Projects/NST/data/style/image.jpeg').to(device)

print("Content Shape",content_img_processed.shape)
print("Style Shape",style_img_processed.shape)

import numpy as np
import matplotlib.pyplot as plt

def deprocess(tensor):
  img = tensor.to('cpu').clone()
  img = img.numpy()   #convert tenser to numpy
  img = img.squeeze(0)  #(1,3,224,224) -> (3.224.224)
  img = img.transpose(1,2,0) #(3,224,224) -> (224,224,3)
  img = img*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])   #image = image*standard_deviation + mean
  img = img.clip(0,1)
  return img

content_img_deprocessed = deprocess(content_img_processed)
style_img_deprocessed = deprocess(style_img_processed)

#fig, (ax1,ax2) = plt.subplots(1,2,figsize = (20,10))
#ax1.imshow(content_img_deprocessed)
#ax2.imshow(style_img_deprocessed)
#ax1.set_xticks([])
#ax1.set_yticks([])
#ax2.set_xticks([])
#ax2.set_yticks([])
#plt.show()

#function to extract style and content features from vgg19 layers
def get_features(image,model):
  layers = {
      '0' : 'conv1_1',
      '5' : 'conv2_1',
      '10' : 'conv3_1',
      '19' : 'conv4_1',
      '21' : 'conv4_2',            #content_feature, rest are for style feature
      '28' : 'conv5_1',
  }
  #now extract features from these layers
  x = image
  #to store our content and style features
  Features = {}

  for name,layer in model._modules.items():
    x = layer(x)      #loading layers from VGG19 via items one by one
    if name in layers:
        Features[layers[name]] = x
  return Features

content_features = get_features(content_img_processed,vgg)
style_features = get_features(style_img_processed,vgg)

def gram_matrix(tensor):
  b,c,h,w = tensor.size()
  tensor = tensor.view(c,h*w)
  gram = torch.mm(tensor,tensor.t())
  return gram

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

def content_loss(target_conv_4_2,content_conv_4_2):
  #mean squared
  loss = torch.mean((target_conv_4_2-content_conv_4_2)**2)
  return loss

style_weights = {
    'conv1_1' : 1.0,
    'conv2_1' : 0.75,
    'conv3_1' : 0.2,
    'conv4_1' : 0.2,
    'conv5_1' : 0.2
}

def style_loss(style_weights,target_features,style_grams):
  loss = 0
  for layer in style_weights:
    target_f = target_features[layer]
    target_gram = gram_matrix(target_f)
    style_gram = style_grams[layer]
    b,c,h,w = target_f.shape
    layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
    loss += layer_loss/(c*h*w)
  return loss

target_img = content_img_processed.clone().requires_grad_(True).to(device)
target_f = get_features(target_img,vgg)

print("Content loss: ",content_loss(target_f['conv4_2'],content_features['conv4_2']))
print("Style loss: ",style_loss(style_weights,target_f,style_grams))

from torch import optim
optimizer = optim.Adam([target_img],lr=0.003)
alpha,beta = 1,1e5
epochs,show_every = 1000,100

def total_loss(c_loss,s_loss,alpha,beta):
  loss = (alpha * c_loss) + (beta * s_loss)
  return loss

results = []

for i in range(epochs):
  target_f = get_features(target_img,vgg)
  c_loss = content_loss(target_f['conv4_2'],content_features['conv4_2'])
  s_loss = style_loss(style_weights,target_f,style_grams)
  t_loss = total_loss(c_loss,s_loss,alpha,beta)

  optimizer.zero_grad()
  t_loss.backward()
  optimizer.step()

  if i % show_every == 0:
    print("Total loss at Epoch {} : {}".format(i,t_loss))
    res = deprocess(target_img.detach())
    results.append(res)


target_copy = deprocess(target_img.detach())
content_copy = deprocess(content_img_processed)

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10,5))
ax1.imshow(target_copy)
ax2.imshow(content_copy)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()

