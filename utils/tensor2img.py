import numpy as np
import torch

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().detach().numpy()
    if normalize:
        image_numpy[0] = (image_numpy[0] * 0.229 + 0.485) * 255
        image_numpy[1] = (image_numpy[1] * 0.224 + 0.456) * 255
        image_numpy[2] = (image_numpy[2] * 0.225 + 0.406) * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def tensor2im2(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().detach().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = image_numpy * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[0] == 23:
        image_numpy = np.concatenate([image_numpy[3:6, :, :], image_numpy[0:3, :, :]], axis=1)
    if image_numpy.shape[0] == 1 or image_numpy.shape[0] > 3:
        image_numpy = image_numpy[0, :,:]
    return image_numpy.astype(imtype)