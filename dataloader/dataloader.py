import os
import numpy as np
import math


class DataLoader():
    def __init__(self):
        self.num_features = 69 # can change this!

        myPath = os.getcwd()
        self.myDataPath = os.path.join(myPath, 'data')
        self.myDataFiles = os.listdir(self.myDataPath)


        # read in data from .npy files
        self.data = {}
        for file in self.myDataFiles:
            posePath = os.path.join(self.myDataPath, file)
            self.data[file[:-5]] = np.load(posePath, allow_pickle=True)
            self.data[file[:-5]].shape = self.data[file[:-5]].shape[0], self.data[file[:-5]].shape[1] * self.data[file[:-5]].shape[2]


    def get_train_test_split(self, train_split = .6, test_split = .2):
        """Returns data in format training_data, testing_data"""

        train_data_size = 0
        test_data_size  = 0
        valid_data_size  = 0

        X_train = None
        X_test  = None
        X_valid = None

        y_train = None
        y_test  = None
        y_valid = None
        
        for i, pose in enumerate(self.data):
            train_class_amount = math.floor(self.data[pose].shape[0] * train_split)
            train_data_size += train_class_amount
            
            test_class_amount = math.floor(self.data[pose].shape[0] * test_split)
            test_data_size  += test_class_amount
            
            valid_class_amount = self.data[pose].shape[0] - train_class_amount - test_class_amount
            valid_data_size  += valid_class_amount
            
            if i == 0:
                X_train = self.data[pose][0:train_class_amount, :]
                X_test = self.data[pose][train_class_amount:train_class_amount+test_class_amount, :]
                X_valid = self.data[pose][train_class_amount+test_class_amount:, :]

                y_train = np.array([i for j in range(train_class_amount)])
                y_test  = np.array([i for j in range(test_class_amount)])
                y_valid = np.array([i for j in range(valid_class_amount)])
            else:
                X_train = np.concatenate((X_train, self.data[pose][0:train_class_amount, :]), axis=0)
                X_test = np.concatenate((X_test, self.data[pose][train_class_amount:train_class_amount+test_class_amount, :]), axis=0)
                X_valid = np.concatenate((X_valid, self.data[pose][train_class_amount+test_class_amount:, :]), axis=0)

                y_train = np.concatenate((y_train, np.array([i for j in range(train_class_amount)])))
                y_test  = np.concatenate((y_test, np.array([i for j in range(test_class_amount)])))
                y_valid  = np.concatenate((y_valid, np.array([i for j in range(valid_class_amount)])))
        
        return X_train, y_train, X_test, y_test, X_valid, y_valid

dl = DataLoader()
X_train, y_train, X_test, y_test, X_valid, y_valid = dl.get_train_test_split()
print("Number of training points = " + str(X_train.shape[0]))
print("Number of testing points = "+ str(X_test.shape[0]))
print("Number of validation points = "+ str(X_valid.shape[0]))