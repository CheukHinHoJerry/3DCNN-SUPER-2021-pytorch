# -*- coding: utf-8 -*-
from OpticalFlow.utils.check_performance import *
from OpticalFlow.utils.load_parameter import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow import keras


def FSRCNN_model(filename, HEIGHT, WIDTH):
    (weights, biases, prelu) = load_parameter_FSRCNN(filename)
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', use_bias=True, input_shape=(HEIGHT, WIDTH, 1),
               trainable=True))

    model.add(Conv2D(8, (1, 1), padding='same', activation='relu', use_bias=True, trainable=True))

    model.add(Conv2D(8, (3, 3), padding='same', activation='relu', use_bias=True, trainable=True))

    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias=True, trainable=True))

    model.add(Conv2DTranspose(16, (4, 4), padding='same', strides=(2, 2), activation='relu', use_bias=True))

    model.add(Conv2D(1, (5, 5), padding='same', activation='relu', use_bias=True))

    for i in range(0, 1):
        model.layers[i].set_weights([weights[i], biases[i]])
        print(model.layers[i].input_shape)
        print(model.layers[i].output_shape)

    print(model.summary())

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0015, decay=7e-9), metrics=[ssim_for2d, psnr_for2d])

    return model


def fsrcnn(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    print("Running FSRCNN model, line before showing image")
    plt.imshow(test_x[0, :, :, 0])
    plt.title('raw test_x')
    plt.savefig('FSRCNN/raw_test.png')
    plt.show()

    # no need to enlarge the train set

    # build model---------------------------------------------------------------------
    filename = '/home/user1/REUS/image-reconstruction-2019-a/papers_and_codes/Accelerating SRCNN/FSRCNN_test/model/FSRCNN-s/x2.mat'
    model = FSRCNN_model(filename, HEIGHT, WIDTH)

    # Train model ------------------------------------------------------------------------
    EPOCHS = 5
    BATCH = 1
    FRACTION = 0.8

    # setting tensorboard
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(train_x, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION, shuffle=True,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', baseline=None,
                                       restore_best_weights=False),
                         ModelCheckpoint(filepath='./FSRCNN/model_climate_fsrcnn.h5', monitor='val_loss', verbose=2,
                                         save_best_only=True, save_weights_only=False, mode='auto', period=1),
                         tensorboard_callback])

    model = keras.models.load_model('./FSRCNN/model_climate_fsrcnn.h5',
                                    custom_objects={'ssim_for2d': ssim_for2d, 'psnr_for2d': psnr_for2d})

    # Make prediction--------------------------------------------------------------------
    final_test = model.predict(test_x)
    final_train = model.predict(train_x)

    result_train_file = h5py.File('climate_data_fsrcnn_result_train.h5', 'w')
    dataset = result_train_file.create_dataset('fsrcnn_processed', data=final_train)
    result_test_file = h5py.File('climate_data_fsrcnn_result_test.h5', 'w')
    dataset = result_test_file.create_dataset('fsrcnn_processed', data=final_test)

    # Performance after fsrcnn model process
    (psnr, ssim, mae, mse) = check_performance(final_train, train_y, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT,
                                               without_padding=False)

    plt.imshow(final_test[0, :, :, 0])
    plt.title('test_x final result')
    plt.savefig('FSRCNN/final_test_x.png')
    plt.imshow(test_y[0, :, :, 0])
    plt.title('test_y')
    plt.savefig('FSRCNN/test_y.png')

    print('psnr_final, ssim_final, mae_final, mse_final')
    print(psnr, ssim, mae, mse, '\n')

    return
