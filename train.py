

from data_loader import DataSet
import Model, metrics
from metrics import RocAucEvaluation
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import time
import os

def main(batch_size, crop_size, random_move, learning_rate,
          weight_decay, save_folder, epochs):
    '''
    :param batch_sizes: the number of examples of each class in a single batch
    :param crop_size: the input size
    :param random_move: the random move in data augmentation
    :param learning_rate: learning rate of the optimizer
    :param segmentation_task_ratio: the weight of segmentation loss in total loss
    :param weight_decay: l2 weight decay
    :param save_folder: where to save the snapshots, tensorflow logs, etc.
    :param epochs: how many epochs to run
    :return:
    '''
    batch_size = batch_size

    train_dataset = DataSet(batch_size=batch_size, crop_size=crop_size, move=random_move,
                                  train_test='train')

    val_dataset = DataSet(batch_size=85, crop_size=crop_size, move=None,
                                train_test='test')

    train_generator = train_dataset.generator()
    val_generator = val_dataset.generator()

    model = Model.get_compiled(output_size=3,
                                    optimizer=Adam(lr=learning_rate),
                                    loss='binary_crossentropy',
                                    metrics=['accuracy'],
                                    
                                    loss_weights=1.,
                                    weight_decay=weight_decay)

    #weight="/home/fubangqi/project/ml/test/4_dense-19_1225_1543/weights/016-0.698.hdf5"
    #model.load_weights(weight)

    time_str = time.strftime("%y_%m%d_%H%M", time.localtime())

    name_str = '3_dense_no_seg' + '-' + time_str

    # Callbacks: Save the model.
    directory = os.path.join(save_folder, name_str)
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory1 = os.path.join(save_folder, name_str, 'weights')
    if not os.path.exists(directory1):
        os.makedirs(directory1)

    directory2 = os.path.join(save_folder, name_str, 'weights')
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    


    checkpointer = ModelCheckpoint(filepath=os.path.join(directory1,'{epoch:03d}-{val_loss:.3f}.hdf5'), verbose=1,
                                   period=1, save_weights_only=False)

    best_keeper = ModelCheckpoint(filepath=os.path.join(directory,'best.h5') , verbose=1, save_weights_only=True,
                                  monitor='val_acc', save_best_only=True, period=1, mode='max')

    csv_logger = CSVLogger(os.path.join(directory, 'training.csv') )

    tensorboard = TensorBoard(log_dir=os.path.join(directory,'logs/') )

    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, mode='max',
                                   patience=30, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.334, patience=5,
                                   verbose=1, mode='max', epsilon=1.e-7, cooldown=2, min_lr=5.e-7)

    RocAuc = RocAucEvaluation(validation_generator=val_generator, interval=1)


    model.fit_generator(generator=train_generator, steps_per_epoch=11, max_queue_size=50, workers=1,
                        validation_data=val_generator, epochs=epochs, validation_steps=1,
                        callbacks=[checkpointer, early_stopping, best_keeper, lr_reducer, csv_logger, tensorboard, RocAuc])


if __name__ == '__main__':
    main(batch_size=32,
         crop_size=[32, 32, 32],
         random_move=3,
         learning_rate=5.e-5,
         weight_decay=0.,
         save_folder='test',
         epochs=30)


'''
                                    metrics=['accuracy', metrics.precision, metrics.recall, metrics.fmeasure,
                                                     metrics.invasion_acc, metrics.invasion_fmeasure,
                                                     metrics.invasion_precision, metrics.invasion_recall,
                                                     metrics.ia_acc, metrics.ia_fmeasure,
                                                     metrics.ia_precision, metrics.ia_recall],
'''