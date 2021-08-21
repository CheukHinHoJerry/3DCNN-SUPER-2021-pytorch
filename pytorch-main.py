import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from mymodel.model_veryDeep2 import Net
from utils.check_performance import *
from utils.load_data_real_video import *
from utils.load_data import *
from sklearn.model_selection import train_test_split
from datetime import date
#from pytorch_model_summary import summary
#from torch.utils.tensorboard import SummaryWriter

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
    loss = mse(predict, target) + alpha*mse(target, aux_output)
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
# Remark : there are total of 360 data in the diamond

# previously using AMOUNT = 240, testing = 245-295

DEPTH = 5
START_INDEX, END_INDEX = 245, 295
AMOUNT = 240
AMOUNT_TEST = END_INDEX - START_INDEX
file_path = "diamond2"

# INDEX indicating index of testing set, where 0 < END_INDEX - START_INDEX < 300, most possible <50 for sufficient training data


(data_x, data_y, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH) = get_input_y_channel(file_path)

# getting triaing data start from initial frame
train_x = data_x[0].reshape((1, 1, HEIGHT, WIDTH))
train_y = data_y[0].reshape((1, 1, TARGET_HEIGHT, TARGET_WIDTH))
for i in range(1, AMOUNT):
    train_x = np.append(train_x, data_x[i].reshape((1, 1, HEIGHT, WIDTH)), axis=0)
    train_y = np.append(train_y, data_y[i].reshape((1, 1, TARGET_HEIGHT, TARGET_WIDTH)), axis=0)

# getting testing set from indicated index
test_x = data_x[START_INDEX].reshape((1, 1, HEIGHT, WIDTH))
test_y = data_y[START_INDEX].reshape((1, 1, TARGET_HEIGHT, TARGET_WIDTH))
for s in range(START_INDEX + 1, END_INDEX):
    test_x = np.append(test_x, data_x[s].reshape((1, 1, HEIGHT, WIDTH)), axis=0)
    test_y = np.append(test_y, data_y[s].reshape((1, 1, TARGET_HEIGHT, TARGET_WIDTH)), axis=0)

print("shape of train_x", train_x.shape)
print("shape of train_y", train_y.shape)
print("shape of test_x", test_x.shape)
print("shape of test_y", test_y.shape)
"""-----------------------------------------------------------------------------------------------------------------"""
plt.imshow(test_x[0, 0, :, :])
plt.title('raw test_x')
# plt.savefig('raw_test_2d3d.png')
plt.show()

plt.imshow(test_y[0, 0, :, :])
plt.title('raw test_y')
# plt.savefig('raw_test_2d3d.png')
plt.show()

"""******************************************Section for chekcing raw error**************************************"""
check_performance_test_x = enlarge(test_x, TARGET_HEIGHT, TARGET_WIDTH)
check_performance_train_x = enlarge(train_x, TARGET_HEIGHT, TARGET_WIDTH)

print("Check error parameters after INTER_NEARST iterpolation")
print('train_psnr_raw, train_ssim_raw, train_mae_raw, train_mse_raw: (on training and validation set)')

(train_psnr_raw, train_ssim_raw, train_mae_raw, train_mse_raw) = check_performance(check_performance_train_x,
                                                                                   train_y, TARGET_HEIGHT,
                                                                                   TARGET_WIDTH, AMOUNT,
                                                                                   without_padding=True)
print(train_psnr_raw, train_ssim_raw, train_mae_raw, train_mse_raw, '\n')


print('psnr_raw, ssim_raw, mae_raw, mse_raw: (on testing set)')
(test_psnr_raw, test_ssim_raw, test_mae_raw, test_mse_raw) = check_performance(check_performance_test_x, test_y,
                                                                               TARGET_HEIGHT, TARGET_WIDTH,
                                                                               AMOUNT_TEST,
                                                                               without_padding=True)
print(test_psnr_raw, test_ssim_raw, test_mae_raw, test_mse_raw, '\n')

"""*********************************************************************************************************************"""

"""******************************************Section for chekcing error after packing**************************************"""

# # cut the middle part of y_set out for ground-truth

train_y_tmp = train_y[2:AMOUNT - 2, :, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2]
train_y = train_y_tmp
test_y_tmp = test_y[2:AMOUNT_TEST - 2, :, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2]
test_y = test_y_tmp

# print("Shape of test_y", test_y.shape)
# check_performance_train_x_packed = repacking(check_performance_train_x)
train_packed = repacking(train_x)
# check_performance_test_x_packed = repacking(
#     check_performance_test_x)  # check_performance_train_packed:(116, 5, 318, 640, 1), train_y:(116, 318, 640, 1)
test_packed = repacking(test_x)

# plt.imshow(test_packed[0, 0, 2, :, :])
# plt.title('test_x after packed')
# plt.savefig('packed_test_2d3d.png')
# plt.show()

# Performance after data re-processing

# (train_psnr_packed, train_ssim_packed, train_mae_packed, train_mse_packed) = check_performance_3d(
#     check_performance_train_x_packed, train_y, TARGET_HEIGHT,
#     TARGET_WIDTH, AMOUNT)

# print('train_psnr_packed, train_ssim_packed, train_mae_packed, train_mse_packed: ')
# print(train_psnr_packed, train_ssim_packed, train_mae_packed, train_mse_packed, '\n')

"""----------------------------------------------------------------------------------------------------------------"""
"""-----------------------------------------------Beginning of model----------------------------------------------"""

# casting data to torch tensors

train_packed = torch.FloatTensor(train_packed)
train_y = torch.FloatTensor(train_y)
test_packed = torch.FloatTensor(test_packed)
test_y = torch.FloatTensor(test_y)

# reshaping tensors for fitting into the network
batch_size = 1
# d1, d2, d3, d4, d5 = train_packed.shape
# train_packed = train_packed.reshape((d1, d2, d5, d3, d4))
# dd1, dd2, dd3, dd4 = train_y.shape
# train_y = train_y.reshape((dd1, dd4, dd2, dd3))

plt.imshow(train_packed[0, 0, 2, :, :])
plt.title('train_x after packed')
# plt.savefig('packed_test_2d3d.png')
plt.show()

final_train_input, final_valid_input, final_train_target, final_valid_target = train_test_split(train_packed, train_y,train_size=0.85)

TrainDataSet = DataSet(final_train_input, final_train_target)
ValidDataSet = DataSet(final_valid_input, final_valid_target)
trainloader = torch.utils.data.DataLoader(TrainDataSet, batch_size=batch_size, num_workers=4, pin_memory=True)
validloader = torch.utils.data.DataLoader(ValidDataSet, batch_size=batch_size, num_workers=4, pin_memory=True)


# testloader = torch.utils.data.DataLoader((test_x,test_y),batch_size = batch_size, num_workers=2)

#writer = SummaryWriter(SummaryWriter('runs/dimand_experiment_1'))

def train(net, patience=10):
    num_of_epoch = 300
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # record loss
            train_loss = loss.item() * len(inputs)
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

                #record  loss
                loss = loss_func(valid_pred, target)
                val_loss = val_loss + loss.item() * len(inputs)
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
            torch.save(net.state_dict(), './mymodel/xsede_saved_veryDeep_again' + str(2) + '.pth')
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
if continueTrain=="YES":
    model_path = input("Input the model path: ")
    net.load_state_dict(torch.load(model_path))

#summary(net, torch.zeros((1, 1, 5, HEIGHT, WIDTH)), show_input=True, print_summary=True, show_hierarchical=False)

if torch.cuda.is_available():
    net = net.to(device)
    print("torch.cuda available, move model to GPU")

# if not specified, patience = 10
train(net, patience=8)