"""
There are two major problems in this procedure

First is add_module, as we can see, the final net called "model" is not a complete VGG Net, because we only preserve 
conv, pool and relu layer, and add content loss and style loss layer, but why is it still working? Because we set
content loss and style loss's forward, return input, so after insertion, it don't affect the previous layer's output and 
the next layer's input. In addition, vgg net provides sequential, so we don't need to rewrite the forward function, we 
only want style loss and content loss. 

Second is LBFGS for content_loss + style_loss, we only need to return the losses combined with retain_variables=True, then
optimizer can update variables automatically. Notice: each iteration will update 20 times.

"""

from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import PIL
from PIL import Image
import  matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

#Cuda parameters
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

#Load Images
imsize = 200 #desired size of the output image

loader = transforms.Compose([transforms.Scale(imsize), #scale imported image
                             transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    #fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


style = image_loader("./images/picasso.jpg").type(dtype)
content = image_loader("./images/dancing.jpg").type(dtype)

assert style.size() == content.size(), "We need to import style and content images of the same size"

unloader = transforms.ToPILImage()

def imshow(tensor):
    image = tensor.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    plt.imshow(image)

fig = plt.figure()

plt.subplot(221)
imshow(style.data)
plt.subplot(222)
imshow(content.data)

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        #we detach the target content from the tree used
        self.target = target.detach() * weight
        #to dynamically compute the gradient: this is a stated value,
        #not a variable. Otherwise the forward method of the criterion will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input * self.weight, self.target)
        self.output = input
        return self.output

    #return the loss each step
    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

class GramMatrix(nn.Module):

        def forward(self, input):
            a, b, c, d = input.size()
            # a = batch size = 1
            # b = number of feature maps
            # (c, d) = dimensions of a feature map (N = c * d)

            features = input.view(a * b, c * d) # resize the tensors

            G = torch.mm(features, features.t())

            #we normalize the values of the gram matrix
            #by dividing by the number of element in each feature maps.
            #Because the longer the vector is, the bigger the Gram matrix is.
            return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram.forward(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion.forward(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss


#Load a pretrained_model
cnn = models.vgg19(pretrained=True).features

if use_cuda:
    cnn = cnn.cuda()


#desired depth layers to compute style/content losses:
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

#just in order to have an iterable access to or list of content/style losses
content_losses = []
style_losses = []

model = nn.Sequential() #the new Sequential module network
gram = GramMatrix() #we need a gram module in order to compute style targets

#move these modules to the GPU if possible.
if use_cuda:
    model = model.cuda()
    gram = gram.cuda()

#weight associated with content and style losses
content_weight = 1
style_weight = 1000

i = 1
for layer in list(cnn):
    if isinstance(layer, nn.Conv2d):
        name = "conv_" + str(i)
        model.add_module(name, layer)

        if name in content_layers:
            #add content loss:
            target = model.forward(content).clone()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            #add style loss:
            target_feature = model.forward(style).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, nn.ReLU):
        name = "relu_" + str(i)
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model.forward(content).clone()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            #add style loss:
            target_feature = model.forward(style).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)


        i += 1

    if isinstance(layer, nn.MaxPool2d):
        name = "pool_" + str(i)
        model.add_module(name, layer)

print("Model: ", model)
#changing maxpool to avgpool
#avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
#model.add_module(name, avgpool)

input = image_loader("./images/dancing.jpg").type(dtype)
#input.data = torch.randn(input.data.size()).type(dtype)

plt.subplot(223)
imshow(input.data)

#gradient descent
input = nn.Parameter(input.data)
optimizer = optim.LBFGS([input])

run = [0]
while run[0] <= 300:

    def closure():
        #correct the values of updated input image
        input.data.clamp_(0, 1)

        optimizer.zero_grad()
        model.forward(input)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.backward()
        for cl in content_losses:
            content_score += cl.backward()

        run[0] += 1
        if(run[0] % 10 == 0):
            print("run" + str(run) + ":")
            print(style_score.data[0])
            print(content_score.data[0])

        return content_score + style_score

    print("out run: " + str(run))
    optimizer.step(closure)

input.data.clamp_(0, 1)

plt.subplot(224)
imshow(input.data)
plt.show()