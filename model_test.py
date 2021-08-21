import matplotlib.pyplot as plt
import torch
import numpy as np
from mymodel.model_veryDeep3 import Net
from utils.check_performance import *
from utils.load_data_real_video import *
from utils.load_data import *
from torch import nn

device = torch.device("cuda")
# Remark : there are total of 300 data in the container data set

DEPTH = 5
START_INDEX, END_INDEX = 310, 350
AMOUNT = 20
AMOUNT_TEST = END_INDEX - START_INDEX
file_path = "diamond2"

# INDEX indicating index of testing set, where 0 < END_INDEX - START_INDEX < 300, most possible <50 for sufficient training data

'''-----------------------uncomment this part if want to prepare the whole data set------------------------------------'''
# TOTAL_AMOUNT_OF_DATA = 300
# (data_x, data_y, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH) = get_input_y_channel(file_path)
# new_data_x = data_x[0].reshape((1, 1, HEIGHT, WIDTH))
# new_data_y = data_y[0].reshape((1, 1, TARGET_HEIGHT, TARGET_WIDTH))
#
# for k in range(1, TOTAL_AMOUNT_OF_DATA):
#     new_data_x = np.append(new_data_x, data_x[k].reshape((1, 1, HEIGHT, WIDTH)), axis=0)
#     new_data_y = np.append(new_data_y, data_y[k].reshape((1, 1, TARGET_HEIGHT, TARGET_WIDTH)), axis=0)
#
# for p in range(2, TOTAL_AMOUNT_OF_DATA - 2):
#     plt.imshow(new_data_x[p, 0, :, :])
#     plt.axis('off')
#     plt.savefig('./data_x_Y_channel/' + str("%03d" % int(p - 2)) + '.png', bbox_inches='tight', pad_inches=0)
#     plt.imshow(new_data_y[p, 0, :, :])
#     plt.axis('off')
#     plt.savefig('./data_y_Y_channel/' + str("%03d" % int(p - 2)) + '.png', bbox_inches='tight', pad_inches=0)
# print("done saving image")
#
# new_data_packed = repacking(new_data_x)
# print(new_data_packed.shape)

'''--------------------------------------------------------------------------------------------------------------------'''
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

print("Check performance of training set when view on a finer scale:", )
(train_psnr_raw, train_ssim_raw, train_mae_raw, train_mse_raw) = check_performance(check_performance_train_x,
                                                                                   train_y, TARGET_HEIGHT,
                                                                                   TARGET_WIDTH, AMOUNT,
                                                                                   without_padding=True)

print('train_psnr_raw, train_ssim_raw, train_mae_raw, train_mse_raw: ')
print(train_psnr_raw, train_ssim_raw, train_mae_raw, train_mse_raw, '\n')

print("Check performance of testing set when view on a finer scale:", )
(test_psnr_raw, test_ssim_raw, test_mae_raw, test_mse_raw) = check_performance(check_performance_test_x, test_y,
                                                                               TARGET_HEIGHT, TARGET_WIDTH,
                                                                               AMOUNT_TEST,
                                                                               without_padding=True)

print("Take the error measurement of testing and save as (parameter)_raw")
print('psnr_raw, ssim_raw, mae_raw, mse_raw: ')
print(test_psnr_raw, test_ssim_raw, test_mae_raw, test_mse_raw, '\n')

"""*********************************************************************************************************************"""

"""******************************************Section for chekcing error after packing**************************************"""

# # cut the middle part of y_set out for ground-truth
train_y_tmp = train_y[2:AMOUNT - 2, :, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2]
train_y = train_y_tmp
test_y_tmp = test_y[2:AMOUNT_TEST - 2, :, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2]
test_y = test_y_tmp

print("Shape of test_y", test_y.shape)
check_performance_train_x_packed = repacking(check_performance_train_x)
train_packed = repacking(train_x)
check_performance_test_x_packed = repacking(
    check_performance_test_x)  # check_performance_train_packed:(116, 5, 318, 640, 1), train_y:(116, 318, 640, 1)
test_packed = repacking(test_x)


plt.imshow(test_packed[0, 0, 2, :, :])
plt.axis('off')
plt.savefig('packed_test_2d3d.png', bbox_inches='tight', pad_inches=0)
plt.show()

# Performance after data re-processing

(train_psnr_packed, train_ssim_packed, train_mae_packed, train_mse_packed) = check_performance_3d(
    check_performance_train_x_packed, train_y, TARGET_HEIGHT,
    TARGET_WIDTH, AMOUNT)

print('train_psnr_packed, train_ssim_packed, train_mae_packed, train_mse_packed: ')
print(train_psnr_packed, train_ssim_packed, train_mae_packed, train_mse_packed, '\n')

(test_psnr_packed, test_ssim_packed, test_mae_packed, test_mse_packed) = check_performance_3d(
    check_performance_test_x_packed, test_y, TARGET_HEIGHT,
    TARGET_WIDTH, AMOUNT_TEST)

print("Take the error measurement of testing and save as (parameter)_packed")
print('psnr_packed, ssim_packed, mae_packed, mse_packed')
print(test_psnr_packed, test_ssim_packed, test_mae_packed, test_mse_packed, '\n')

model_path = 'mymodel/xsede_saved_veryDeep3.pth'


def testOneSample(train_packed, test_packed, train_y, test_y, type):
    with torch.no_grad():
        model = Net()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # convert all data to FloatTensor
        train_packed = torch.FloatTensor(train_packed)
        train_y = torch.FloatTensor(train_y)
        test_packed = torch.FloatTensor(test_packed)
        test_y = torch.FloatTensor(test_y)
        mini_test = test_packed[15, :, :, :, :]
        mini_test_target = test_y[15, :, :, :]
        torch.reshape(mini_test, (1, 5, 1, 150, 225))
        mini_test = mini_test.unsqueeze_(0)
        print(mini_test.shape)

        if type != "ONE":
            prediction, aux_output = model(mini_test)
        else:
            prediction = model(mini_test)

        prediction = np.array(prediction)
        mini_test = np.array(mini_test)
        mini_test_target = np.array(mini_test_target)

        print(prediction.shape)
        print(mini_test.shape)
        print(mini_test_target.shape)
        print(np.amax(prediction))
        np.savetxt('prediction', prediction[0, 0, :, :].astype(int), delimiter=',', fmt='%d')
        np.savetxt('input', mini_test[0, 0, 2, :, :].astype(int), delimiter=',', fmt='%d')
        np.savetxt('groundturth', mini_test_target[0, :, :].astype(int), delimiter=',', fmt='%d')
        np.savetxt('difference', prediction[0, 0, :, :] - mini_test_target[0, :, :], delimiter=',', fmt='%d')
        print("max prediction", np.max(prediction))
        print("min prediction", np.min(prediction))
        print("max input", np.max(mini_test))
        print("min input", np.min(mini_test))
        print("max target", np.max(mini_test_target))
        print("min target", np.min(mini_test_target))

        plt.imshow(mini_test[0, 0, 2, :, :])
        plt.title('mini test input')
        plt.savefig('mini_test_input.png')
        plt.show()

        plt.imshow(mini_test_target[0, :, :])
        plt.title('mini test gound truth')
        plt.savefig('mini_test_ground.png')
        plt.show()

        plt.imshow(prediction[0, 0, :, :])
        plt.title('prediction')
        plt.savefig('mini_test_prediciton.png')
        plt.show()
        return


def testWholeTestingSet(train_packed, test_packed, train_y, test_y, type):
    loss_func = nn.MSELoss()
    with torch.no_grad():
        model = Net()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # convert all data to FloatTensor
        train_packed = torch.FloatTensor(train_packed)
        train_y = torch.FloatTensor(train_y)
        test_packed = torch.FloatTensor(test_packed)
        test_y = torch.FloatTensor(test_y)

        if type != "ONE":
            train_prediction, train_aux_output = model(train_packed)
            test_prediction, test_aux_output = model(test_packed)
        else:
            train_prediction = model(train_packed)
            test_prediction = model(test_packed)

        train_loss = loss_func(train_prediction, train_y)
        test_loss = loss_func(test_prediction, test_y)

        training_psnr, training_ssim = PSNR_SSIM_pytorch(train_prediction, train_y)
        testing_psnr, testing_ssim = PSNR_SSIM_pytorch(test_prediction, test_y)

        print("train_loss: ", train_loss.item())
        print("test_loss: ", test_loss.item())
        print("training psnr: ", training_psnr)
        print("training ssim: ", training_ssim)
        print("testing psnr: ", testing_psnr)
        print("testing ssim: ", testing_ssim)

        return


def saveAllFig(train_packed, test_packed, train_y, test_y, type):
    with torch.no_grad():
        model = Net()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # convert all data to FloatTensor
        train_packed = torch.FloatTensor(train_packed)
        train_y = torch.FloatTensor(train_y)
        test_packed = torch.FloatTensor(test_packed)
        test_y = torch.FloatTensor(test_y)

    plt.imshow(test_packed[0, 0, 2, :, :])
    plt.axis('off')
    plt.savefig('packed_test_2d3d.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def makeWholeData(data_packed):
    with torch.no_grad():
        model = Net()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # convert all data to FloatTensor
        data_packed = torch.FloatTensor(data_packed)
        print(data_packed.shape)
        print("Finish predicting data, now proceed to save image")

        for p in range(TOTAL_AMOUNT_OF_DATA):
            print(data_packed[p, :, :, :, :].shape)
            current_input = torch.reshape(data_packed[p, :, :, :, :], (1, 1, 5, 150, 225))
            # current_input = current_input.unsqueeze_(0)
            # print(current_input.shape)
            data_prediction = model(current_input)
            print(data_prediction.shape)
            plt.imshow(data_prediction[0, 0, :, :])
            plt.axis('off')
            plt.savefig('./wholeset_prediction/' + str("%03d" % p) + '.png', bbox_inches='tight', pad_inches=0)
            if p < 3:
                plt.show()
            print("finished", p)
        return


type = input("Enter ONE for model without 2d loss, enter any other input model with 2d loss: ")
choice = input("Enter ONE for single image, enter any other input for testing set: ")
if choice == "ONE":
    testOneSample(train_packed, test_packed, train_y, test_y, type)
else:
    testWholeTestingSet(train_packed, test_packed, train_y, test_y, type)

# makeWholeData(new_data_packed)
