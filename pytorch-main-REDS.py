import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model1 import Net
from utils.check_performance import *
from utils.load_data_real_video import *
from utils.load_data import *
from sklearn.model_selection import train_test_split
from datetime import date
from pytorch_model_summary import summary

# from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataSet_x, dataSet_y):
        self.dataSet_x = dataSet_x
        self.dataSet_y = dataSet_y

    def __len__(self):
        return len(self.dataSet_x)

    def __getitem__(self, index):
        return (self.dataSet_x[index], self.dataSet_y[index])


def control2DLoss(predict, target, aux_output):
    alpha = 0.00
    mse = nn.MSELoss()
    aux_output = torch.reshape(aux_output, target.shape)
    loss = mse(predict, target) + alpha * mse(target, aux_output)
    return loss


"""----------------------------------- When training with climate data, uncomment this part-------------------------"""

# DATA_FILE = 'climate_data.h5'
# START_INDEX, END_INDEX = 650, 700
# AMOUNT = 120
# CHANNEL = 1
# DEPTH = 5
#
# # notice that the AMOUNT is the AMOUNT of the training set, but not training set + testing set
# (train_x, AMOUNT, HEIGHT, WIDTH) = load_h5_data(DATA_FILE, 'train', AMOUNT)
# (train_y, AMOUNT, TARGET_HEIGHT, TARGET_WIDTH) = load_h5_data(DATA_FILE, 'label', AMOUNT)
#
# (test_x, AMOUNT_TEST, HEIGHT, WIDTH) = load_h5_data_testset(DATA_FILE, 'train', START_INDEX, END_INDEX)
# (test_y, AMOUNT_TEST, TARGET_HEIGHT, TARGET_WIDTH) = load_h5_data_testset(DATA_FILE, 'label', START_INDEX, END_INDEX)
#
# print(train_x.shape)
# print(train_y.shape)
# 'HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH:', 159 320 318 640

"""----------------------------------------------------------------------------------------------------------------"""

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    devide = torch.device("cpu")
    print("Running on CPU")
""" --------------------------When training with other data sets, uncomment this part-------------------------------"""

DEPTH = 5

file_path = "REDS"
batch_size = 1
# INDEX indicating index of testing set, where 0 < END_INDEX - START_INDEX < 300, most possible <50 for sufficient training data

print("preparing training data and validation data")
(extract_train_x, extract_train_y, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH) = get_input_y_channel_REDS_train(
    file_path)
(extract_valid_x, extract_valid_y, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH) = get_input_y_channel_REDS_valid(
    file_path)

train_sample = extract_train_x[0, 0, 2, :, :]
print(extract_train_x.shape)
plt.imshow(train_sample)
plt.title('test_x after packed')
plt.savefig('packed_test_2d3d.png')
plt.show()

TrainDataSet = DataSet(extract_train_x, extract_train_y)
ValidDataSet = DataSet(extract_valid_x, extract_valid_y)
trainloader = torch.utils.data.DataLoader(TrainDataSet, batch_size=batch_size, num_workers=4, pin_memory=True)
validloader = torch.utils.data.DataLoader(ValidDataSet, batch_size=batch_size, num_workers=4, pin_memory=True)
batch_size = 1


# testloader = torch.utils.data.DataLoader((test_x,test_y),batch_size = batch_size, num_workers=2)

# writer = SummaryWriter()

def train(net, patience=10):
    num_of_epoch = 1000
    min_val_loss = np.inf
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_of_epoch):
        time_start.record()
        epoch_loss = 0
        training_psnr_epoch = 0
        training_ssim_epoch = 0
        val_psnr_epoch = 0
        val_ssim_epoch = 0
        stop = 0
        for data in trainloader:
            inputs, target = data
            if torch.cuda.is_available():
                inputs, target = inputs.to(device), target.to(device)
            # Forward Pass
            with torch.cuda.amp.autocast():
                pred = net(inputs)
                assert pred.dtype is torch.float16
                # Find the Loss
                loss = loss_func(pred, target)
                assert loss.dtype is torch.float32
            '''----------------------------------------------------normal grad ------------------------------------'''
            # # Clear grad
            # optimizer.zero_grad()
            # # backward propagation
            # loss.backward()
            # # update
            # optimizer.step()
            '''---------------------------------------------------------------------------------------------------'''
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            optimizer.zero_grad()
            # Calculate Loss of each batch
            train_loss = loss.item() * len(data)
            # Save loss for whole epoch
            epoch_loss += train_loss
            training_psnr, training_ssim = PSNR_SSIM_pytorch(pred, target)
            training_psnr_epoch += training_psnr
            training_ssim_epoch += training_ssim
        val_loss = 0
        net.eval()  # Optional when not using Model Specific layer

        for data in validloader:
            inputs, target = data
            if torch.cuda.is_available():
                inputs, target = inputs.to(device), target.to(device)
            # Forward Pass
            with torch.no_grad():
                valid_pred = net(inputs).to(device)
                # Find the Loss
                loss = loss_func(valid_pred, target)
                # Calculate Loss
                val_loss = val_loss + loss.item() * len(data)
                val_psnr, val_ssim = PSNR_SSIM_pytorch(valid_pred, target)
                val_psnr_epoch += val_psnr
                val_ssim_epoch += val_ssim

        print("Epoch", epoch + 1)
        print("Training loss: ", epoch_loss / len(trainloader), "Validation Loss: ",
              val_loss / len(validloader))
        print("Training psnr: ", training_psnr_epoch / len(trainloader), "Validation psnr: ",
              val_psnr_epoch / len(validloader))
        print("Training ssim: ", training_ssim_epoch / len(trainloader), "Validation ssim: ",
              val_ssim_epoch / len(validloader))

        if min_val_loss > val_loss:
            print(
                f'__Validation Loss Decreased({min_val_loss / len(validloader):.6f}--->{val_loss / len(validloader):.6f}) \t Saving The Model__')
            min_val_loss = val_loss
            # Saving State Dict
            torch.save(net.state_dict(), './mymodel/saved_model_add_across_net' + str(1) + '.pth')
        else:
            stop += 1
            if stop >= patience:
                print("early stop")
                return

        time_end.record()
        print("Finished Epoch", epoch + 1, '--', math.floor(time_start.elapsed_time(time_end) / 1000), 's')
        print("")


net = Net()
continueTrain = input("Do you want to continue training? Type YES if continue")
if continueTrain == "YES":
    model_path = input("Input the model path")
    net.load_state_dict(torch.load(model_path))

summary(net, torch.zeros((1, 1, 5, HEIGHT, WIDTH)), show_input=True, print_summary=True, show_hierarchical=False)

if torch.cuda.is_available():
    net = net.to(device)
    print("torch.cuda available, move model to GPU")

# if not specified, patience = 10
train(net, patience=8)
