import numpy
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import torch
from torch.nn import MSELoss,L1Loss
# PSNR and SSIM calculation for inputting a 3d array (amount, height, width)

def PSNR_SSIM(y, x):
    (AMT, HT, WDTH) = y.shape
    PSNR_sum = 0
    SSIM_sum = 0

    for i in range(AMT):
        # reshaping a 3d matrix into a 2d matrix with same height and width
        y_tmp = y[i, :, :].reshape((HT, WDTH))
        x_tmp = x[i, :, :].reshape((HT, WDTH))

        # for climate data
        # PSNR_sum += peak_signal_noise_ratio(x_tmp, y_tmp, data_range=1.0)
        # SSIM_sum += ssim(x_tmp, y_tmp, data_range=1.0)

        # for container data
        PSNR_sum += peak_signal_noise_ratio(x_tmp, y_tmp, data_range=255.0)
        SSIM_sum += ssim(x_tmp, y_tmp, data_range=255.0)

    PSNR_sum = PSNR_sum / AMT
    SSIM_sum = SSIM_sum / AMT

    return (PSNR_sum, SSIM_sum)


# MAE and MSE calculation for inputting a 3d array (amount, height, width)
def PSNR_SSIM_pytorch(x, y):
    (AMT, CHANNEL, HT, WDTH) = y.shape
    PSNR_sum = 0
    SSIM_sum = 0
    # casting back to np array for calculation
    with torch.no_grad():
        y = np.array(torch.Tensor.cpu(y))
        x = np.array(torch.Tensor.cpu(x))
        for i in range(AMT):
            # reshaping a 4d matrix into a 2d matrix with same height and width
            y_tmp = y[i, 0, :, :].reshape((HT, WDTH))
            x_tmp = x[i, 0, :, :].reshape((HT, WDTH))

            # for drange = 1
            # PSNR_sum += peak_signal_noise_ratio(x_tmp, y_tmp, data_range=1.0)
            # SSIM_sum += ssim(x_tmp, y_tmp, data_range=1.0)

            # for drange = 255
            PSNR_sum += peak_signal_noise_ratio(x_tmp, y_tmp, data_range=255.0)
            SSIM_sum += ssim(x_tmp, y_tmp, data_range=255.0)

    PSNR_sum = PSNR_sum / AMT
    SSIM_sum = SSIM_sum / AMT

    return (PSNR_sum, SSIM_sum)


def MAE_MSE(y, x):
    loss = MSELoss()
    l1loss = L1Loss()
    (AMT, HT, WDTH) = y.shape
    mae = 0
    mse = 0
    for i in range(AMT):
        # reshaping a 3d matrix into a 2d matrix with same height and width
        y_tmp = y[i, :, :].reshape((HT, WDTH))
        x_tmp = x[i, :, :].reshape((HT, WDTH))
        x_tmp = torch.FloatTensor(x_tmp)
        y_tmp = torch.FloatTensor(y_tmp)
        with torch.no_grad():
            mae += l1loss(y_tmp, x_tmp).item()
            mse += loss(y_tmp, x_tmp).item()

    mae = mae / AMT
    mse = mse / AMT

    return (mae, mse)


# check the performance given two set of array with specified height, width and amount. If they

def check_performance(x_set, y_set, HEIGHT, WIDTH, AMOUNT, without_padding):
    # only for x and y in the same shape
    if without_padding == True:
        x = numpy.reshape(x_set[:, :, 2:HEIGHT - 2, 2:WIDTH - 2], (AMOUNT, HEIGHT - 2 * 2, WIDTH - 2 * 2))
        y = numpy.reshape(y_set[:, :, 2:HEIGHT - 2, 2:WIDTH - 2], (AMOUNT, HEIGHT - 2 * 2, WIDTH - 2 * 2))
    else:
        x = numpy.reshape(x_set[:, :, :, :], (AMOUNT, HEIGHT, WIDTH))
        y = numpy.reshape(y_set[:, :, :, :], (AMOUNT, HEIGHT, WIDTH))
    (psnr, ssim) = PSNR_SSIM(y, x)
    (mae, mse) = MAE_MSE(y, x)

    return (psnr, ssim, mae,mse)


def check_performance_3d(x_set, y_set, HEIGHT, WIDTH, AMOUNT):
    x = numpy.reshape(x_set[:, :, 2, 2:HEIGHT - 2, 2:WIDTH - 2], (AMOUNT - 4, HEIGHT - 2 * 2, WIDTH - 2 * 2))
    y = numpy.reshape(y_set[:, :, :, :], (AMOUNT - 4, HEIGHT - 2 * 2, WIDTH - 2 * 2))
    (psnr, ssim) = PSNR_SSIM(y, x)
    (mae, mse) = MAE_MSE(y, x)
    return (psnr, ssim, mae, mse)
