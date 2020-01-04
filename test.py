
from data_loader import DataSet
import Model, metrics
import time
import os
import numpy as np
import csv

val_dataset = DataSet(batch_size=1, crop_size=32, move=None,
                                train_test='test')

val_generator = val_dataset.test_generator()

#weights="/home/fubangqi/project/ml/test/test-19_1216_0957/weights/029-0.641.hdf5"
#weights="/home/fubangqi/project/ml/test/test-19_1222_1007/weights/050-0.678.hdf5"
#weights="/home/fubangqi/project/ml/test/test-19_1223_1123/weights/024-0.706.hdf5"
#weights="/home/fubangqi/project/ml/test/test-19_1223_1214/weights/006-1.235.hdf5"
#weights="/home/fubangqi/project/ml/test/test-19_1223_1214/weights/027-0.770.hdf5"
#weights="/home/fubangqi/project/ml/test/test-19_1223_1247/weights/023-0.655.hdf5"
#weights="/home/fubangqi/project/ml/test/re_test-19_1224_0746/weights/013-0.699.hdf5"
#weights="/home/fubangqi/project/ml/test/re_test-19_1224_0746/weights/030-0.664.hdf5"
weights="./weights/014-0.699.hdf5" #0.660
#weights="/home/fubangqi/project/ml/test/3dense_seg_norm-19_1224_1216/weights/049-0.882.hdf5"
#weights="/home/fubangqi/project/ml/test/re-19_1225_1232/weights/025-1.117.hdf5"   #merry 0.8+0.2
#weights="/home/fubangqi/project/ml/test/re-19_1225_1155/weights/010-0.882.hdf5"
#weights="/home/fubangqi/project/ml/test/2nd_test_re-19_1225_1431/weights/001-0.657.hdf5"
#weights="/home/fubangqi/project/ml/test/4_dense-19_1225_1543/weights/016-0.698.hdf5"
#weights="/home/fubangqi/project/ml/test/4_dense-19_1225_1535/weights/003-0.651.hdf5"
#weights="/home/fubangqi/project/ml/test/3_dense_no_seg-19_1225_1647/weights/004-0.674.hdf5"
model = Model.get_model(weights=weights)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])




headers=['Id','Predicted']
l=[]
for i in range(117):
    a,_,c=next(val_generator)
    preds=model.predict(a)
    #result = np.argmax(preds, axis = 1)
    l.append((c[0], preds[0][0]))
path='./submit.csv'

p=open(path,'w')
writer = csv.writer(p)
writer.writerow(headers)
writer.writerows(l)
p.close()
