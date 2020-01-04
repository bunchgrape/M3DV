"""
Class for managing our data.
"""
import csv
import numpy as np
import os.path
import threading
import random
from utils import rotation, reflection, crop, random_center, _triple
from keras.utils import to_categorical

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen



class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret



class DataSet():
    def __init__(self, batch_size=32, crop_size=32, move=None, train_test='train'):
        """Constructor.
        opt_flow_len = (int) the number of optical flow frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.move = move
        self.train_test = train_test
        # Get the data.
        self.data_list, self.data_list_val = self.get_data_list()

        self.test_list = self.get_test()

        self.num_train = len(self.data_list)
        self.num_val = len(self.data_list_val)
        self.num_test = len(self.test_list)

        self.train_path = "./data/train_val"
        self.test_path = "./data/test"
        #self.train_path = "E:/workspace/ml/train_val"
        
        self.transform = Transform(crop_size, move)
        self.transform_test = Transform(crop_size, None)

    @staticmethod
    def get_data_list():
        """Load our data list from file."""
        with open(os.path.join('./data', 'train_val.csv'), 'r') as fin:
        #with open(os.path.join('E:/workspace/ml', 'train_val.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data_list = list(reader)

        data_list_train = data_list[1:381]
        data_list_val = data_list[381:]
        return data_list_train, data_list_val

    @staticmethod
    def get_test():

        with open(os.path.join('./data', 'sampleSubmission.csv'), 'r') as fin:
        #with open(os.path.join('E:/workspace/ml', 'train_val.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data_list = list(reader)
        return data_list[1:]
        
    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""

        # Encode it first.
        
        # Now one-hot it.
        y = to_categorical(class_str, len([0, 1]))


        return y

    @threadsafe_generator
    def generator(self):
        """Return a generator of optical frame stacks that we can use to test."""

        #print("\nCreating validation generator with %d samples.\n" % len(self.data_list))

        if (self.train_test == 'train'):
            data_list = self.data_list
            num_sample = self.num_train
        elif (self.train_test == 'test'):
            data_list = self.data_list_val
            num_sample = self.num_val

        idx = 0
        while 1:
            
            X_batch = []
            y_batch = []

            name_batch = []

            # Get a list of batch-size samples.

            #for row in batch_list:
            start = idx
            for i in range(self.batch_size):
                # Get the stacked optical flows from disk.
                if i+start == num_sample:
                    start = 0
                    random.shuffle(data_list)

                row  = data_list[i+start]
                name = row[0]
                #label = self.get_class_one_hot(row[1])
                with np.load(os.path.join(self.train_path, '%s.npz' % name)) as npz:
                    voxel=npz['voxel']
                    #seg=npz['seg']
                    #seg=(npz['seg'] * 0.8 + 0.2)
                    #voxel=seg*voxel
                    voxel = self.transform(voxel)
                X_batch.append(voxel)
                y_batch.append(row[1])
                name_batch.append(name)
                # Get the corresponding labels

            idx += self.batch_size
            idx = idx % num_sample
            
            #print("\nGenerating batch number {0}/{1} ...".format(idx + 1, n_batch))
            

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            name_batch = np.array(name_batch)
            yield X_batch, y_batch

    @threadsafe_generator
    def test_generator(self):
        """Return a generator of optical frame stacks that we can use to test."""

        #print("\nCreating validation generator with %d samples.\n" % len(self.data_list))
        data_list = self.test_list
        num_sample= self.num_test
        idx = 0
        while 1:
            X_batch = []
            y_batch = []

            name_batch = []
            start = idx
            for i in range(self.batch_size):
                # Get the stacked optical flows from disk.
                if i+start == num_sample:
                    start = 0

                row  = data_list[i+start]
                name = row[0]
                
                with np.load(os.path.join(self.test_path, '%s.npz' % name)) as npz:
                    voxel=npz['voxel']
                    #seg=npz['seg']
                    #seg=(npz['seg'] * 0.8 + 0.2)
                    #voxel=seg*voxel
                    voxel = self.transform_test(voxel)
                X_batch.append(voxel)

                name_batch.append(name)
                # Get the corresponding labels

            idx += self.batch_size
            idx = idx % num_sample
    
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            name_batch = np.array(name_batch)
            yield X_batch, y_batch, name_batch

