import copy
import numpy as np
import cv2
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import matplotlib.pyplot as plt
import pysaliency
import deepgaze_pytorch
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
DEVICE = 'cpu'
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
centerbias_template = np.load('centerbias_mit1003.npy')


def deepgaze(img):
    image = img
    origin = copy.deepcopy(image)
    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.

    # rescale to match image size
    centerbias = zoom(centerbias_template,
                      (image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]),
                      order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor)

    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    b, g, r = cv2.split(origin)
    origin = cv2.merge((r, g, b))
    axs[0].imshow(origin)
    # axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
    # axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
    axs[0].set_axis_off()
    axs[1].matshow(
        log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
    # axs[1].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
    # axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
    axs[1].set_axis_off()
    plt.show()


if __name__ == "__main__":
    img = cv2.imread('../DeepGaze-main/images/3.jpg')
    deepgaze(img)
    dataset_location = 'datasets'
    model_location = 'models'



# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from scipy.misc import face
# from scipy.ndimage import zoom
# from scipy.special import logsumexp
# import torch
#
# import deepgaze_pytorch
#
# DEVICE = 'cpu'
#
# # you can use DeepGazeI or DeepGazeIIE
# model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)
#
# image = cv2.imread('../DeepGaze-main/image/pic1.jpg')
#
# # location of previous scanpath fixations in x and y (pixel coordinates), starting with the initial fixation on the image.
# fixation_history_x = np.array([1024//2, 300, 500, 200, 200, 700])
# fixation_history_y = np.array([768//2, 300, 100, 300, 100, 500])
#
# # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
# # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
# # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
# centerbias_template = np.load('centerbias_mit1003.npy')
# # rescale to match image size
# centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
# # renormalize log density
# centerbias -= logsumexp(centerbias)
#
# image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
# centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
# x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
# y_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
#
# log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
#
# f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
# axs[0].imshow(image)
# axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
# axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
# axs[0].set_axis_off()
# axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
# axs[1].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
# axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
# axs[1].set_axis_off()
#
# plt.show()