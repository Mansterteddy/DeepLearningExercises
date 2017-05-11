# Predict the image's class
import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data

import torchvision.transforms as transforms

# Classes
classes = ["Boy, short", "Girl, short", "Boy, bold", "Boy, medium", "Girl, long"]

# Load net
print("Finding checkpoint")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
use_cuda = torch.cuda.is_available() #check cuda
if use_cuda:
    checkpoint = torch.load('./checkpoint/ckpt.t7')
else:
    # Load GPU version Net to CPU version Net
    checkpoint = torch.load('./checkpoint/ckpt.t7', map_location=lambda storage, location: storage)
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

print "best acc: ", best_acc

# Load Test Data

test_path = "./check"

class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('JPG'):
                images.append('{}'.format(filename))
        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_loader = data.DataLoader(TestImageFolder(test_path, transforms.Compose([transforms.Scale(256),
                                                                             transforms.CenterCrop(224),
                                                                             transforms.ToTensor(),
                                                                             normalize,])),
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,)

# Run the net

net.eval()

print len(test_loader)

for i, (images, filepath) in enumerate(test_loader):

    print "filepath: ", filepath
    image_var = torch.autograd.Variable(images, volatile=True)
    y_pred = net(image_var)

    # get the index of the max log-probability

    print "y_pred: ", classes[np.argmax(y_pred.data[0].numpy())]


'''
def test(test_loader, model):
    csv_map = {}
    # switch to evaluate mode
    model.eval()
    for i, (images, filepath) in enumerate(test_loader):
        # pop extension, treat as id to map
        filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
        filepath = int(filepath)

        image_var = torch.autograd.Variable(images, volatile=True)
        y_pred = model(image_var)
        # get the index of the max log-probability
        smax = nn.Softmax()
        smax_out = smax(y_pred)[0]
        cat_prob = smax_out.data[0]
        dog_prob = smax_out.data[1]
        prob = dog_prob
        if cat_prob > dog_prob:
            prob = 1 - cat_prob
        prob = np.around(prob, decimals=4)
        prob = np.clip(prob, .0001, .999)
        csv_map[filepath] = prob
        # print("{},{}".format(filepath, prob))

    with open(os.path.join(args.data, 'entry.csv'), 'wb') as csvfile:
        fieldnames = ['id', 'label']
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'label'))
        for row in sorted(csv_map.items()):
            csv_w.writerow(row)

    return

'''


















