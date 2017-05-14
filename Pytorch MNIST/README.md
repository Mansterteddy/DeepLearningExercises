# PyTorch MNIST Handler

torchvision provides a good interface to handle datasets online, you can download the datasets automatically. But sometimes, using automate download is too slow, PyTorch allows you download manually, and add the dataset to the directory. But for MNIST datasets, you should transform the download files to training.pt and test.pt, you can use this python file to convert the download data to training.pt and test.pt. The procedure is as follows:

1. Download MNIST datasets from LeCun's websites.
2. Unzip the datasets,
3. Run py3_MNIST_Handler.py.

