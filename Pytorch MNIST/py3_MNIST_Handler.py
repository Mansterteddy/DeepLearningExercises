from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np

urls = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]
raw_folder = 'raw'
processed_folder = 'processed'
training_file = 'training.pt'
test_file = 'test.pt'

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)


'''
for url in self.urls:
    print('Downloading ' + url)
    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    file_path = os.path.join(self.root, self.raw_folder, filename)
    with open(file_path, 'wb') as f:
        f.write(data.read())
    with open(file_path.replace('.gz', ''), 'wb') as out_f, \
            gzip.GzipFile(file_path) as zip_f:
        out_f.write(zip_f.read())
    os.unlink(file_path)'''

# process and save as torch files
print('Processing')

training_set = (
    read_image_file(os.path.join("../data/mnist/processed", 'train-images-idx3-ubyte')),
    read_label_file(os.path.join("../data/mnist/processed", 'train-labels-idx1-ubyte'))
)
test_set = (
    read_image_file(os.path.join("../data/mnist/processed", 't10k-images-idx3-ubyte')),
    read_label_file(os.path.join("../data/mnist/processed", 't10k-labels-idx1-ubyte'))
)


with open(os.path.join("../data/mnist/processed/training.pt"), 'wb') as f:
    torch.save(training_set, f)
with open(os.path.join("../data/mnist/processed/test.pt"), 'wb') as f:
    torch.save(test_set, f)

print('Done!')