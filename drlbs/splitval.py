import mat73
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

data_path = 'indian_pines_randomSampling_0.1_run_1.mat'

dataset = mat73.loadmat(data_path)

X=dataset['x_tra']
# print(X.dtype, X.shape)
y=dataset['y_tra']
# print(y.dtype, y.shape)
X_test=dataset['x_test']
y_test=dataset['y_test']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)
# print(X_train.shape,X_val.shape)

dataset1 = dict({'x_train':X_train,'y_train':y_train, 'x_val':X_val, 'y_val':y_val,'x_test':X_test, 'y_test':y_test})
# print(dataset1)
with open('indiapinedataset.pkl', 'wb') as handle:
    pickle.dump(dataset1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('indiapinedataset.pkl', 'rb') as handle:
    b = pickle.load(handle)

# print(b)

