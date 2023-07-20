import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import numpy as np


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imsave_numpy(img, path):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    # print(npimg.shape)
    plt.imshow(npimg)
    plt.savefig(path)


def imsave_pil(img, path):
    transforms.ToPILImage(mode="RGB")(img).save(path)


def imsave_torch(img, path):
    torchvision.utils.save_image(img, path)