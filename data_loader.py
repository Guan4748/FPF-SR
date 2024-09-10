import math
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        mms = MinMaxScaler(feature_range=(0, 1))
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            mms.fit_transform(train_data.values)
            data = mms.fit_transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        mms = MinMaxScaler(feature_range=(0, 1))
        return mms.fit_transform(data.cpu())


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Covid(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.dropna()

        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        num_train = int(len(df_raw) * (0.6 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        ## min max scaler
        mms = MinMaxScaler(feature_range=(0, 1))
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            mms.fit_transform(train_data.values)
            data = mms.fit_transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        mms = MinMaxScaler(feature_range=(0, 1))
        return mms.fit_transform(data.cpu())


# min max scaler
class Dataset_Custom_(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.dropna()

        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        ## min max scaler
        mms = MinMaxScaler(feature_range=(0, 1))
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            mms.fit_transform(train_data.values)
            data = mms.fit_transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # data_stamp = df_stamp.drop(['date'], 1).values
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        a = seq_x.tolist()
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        mms = MinMaxScaler(feature_range=(0, 1))
        return mms.fit_transform(data.cpu())
        # return self+
        #
        #
        #
        #
        #
        #
        #
        #
        # .scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h', train_only=False):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.dropna()

        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


import pickle


class Dataset_Weibo(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='weibo.pkl',
                 target='weibo', scale=True, timeenc=0, freq='h', train_only=False, data_name = 'hawkes', number=15,
                 seq_len=120, label_len=105,pred_len=106, enc_number = 10) :
        # size [seq_len, label_len, pred_len]
        # info

        self.number = number
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_number = enc_number -1

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.data_name = data_name

    def __read_data__(self):
        self.scaler = StandardScaler()

        # #单步预测
        if self.set_type == 0:
            # 旧的
            # # if self.data_name == 'hawkes':
            # with open("./dataset/haoyu/weibo_train.pkl", 'rb') as f0:
            #     data = pickle.load(f0)
            # with open("./dataset/weibo_tgt_train_1.pkl", 'rb') as f0:

            # 16-----------------
            # with open("./dataset/hawkes/data_train_hawkes.pkl", 'rb') as f0:
            #     data = pickle.load(f0)
            # with open("./dataset/hawkes/data_train_hawkes_related_20.pkl", 'rb') as f0:
            #     data_related = pickle.load(f0)
            # # 时间标签-------------
            with open("./dataset/weibo_tgt_train_time.pkl", 'rb') as f0:
                data_time = pickle.load(f0)

            # # twitter -------------
            # with open("./dataset/twitter/Twitter_train_20_to_Frets.pkl", 'rb') as f0:
            #     twitter_data = pickle.load(f0)
            # # weibo22---------
            # with open("./dataset/weibo22/weibo22_train_20_to_Frets.pkl", 'rb') as f0:
            #     weibo22_data = pickle.load(f0)
            with open("./dataset/weibo21/weibo21_train_20_to_Frets.pkl", 'rb') as f0:
                weibo21_data = pickle.load(f0)

            # Base——————————————
            # with open("./dataset/Base_data/21/21_train_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_train_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_train_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_train_manhattan_20.pkl", 'rb') as f0:
            #     weibo21_data = pickle.load(f0)
            # 22Base——————————————
            # with open("./dataset/Base_data/22/22_train_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_train_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_train_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_train_manhattan_20.pkl", 'rb') as f0:
            #     weibo22_data = pickle.load(f0)
            # twitter Base——————————————
            # with open("./dataset/Base_data/twitter/twitter_train_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_train_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_train_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_train_manhattan_20.pkl", 'rb') as f0:
            #     twitter_data = pickle.load(f0)

            # 16 Base——————————————
            # with open("./dataset/Base_data/16/16_train_cosine_20.pkl", 'rb') as f0:
                # with open("./dataset/Base_data/16/16_train_euclidean_20.pkl", 'rb') as f0:
                # with open("./dataset/Base_data/16/16_train_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/16/16_train_manhattan_20.pkl", 'rb') as f0:
            #     weibo16_data = pickle.load(f0)

        if self.set_type == 1:
            # 旧的
            # with open("./dataset/haoyu/weibo_valid.pkl", 'rb') as f0:
            # with open("./dataset/weibo_tgt_valid_1.pkl", 'rb') as f0:

            # 16-----------------
            # with open("./dataset/hawkes/data_val_hawkes.pkl", 'rb') as f0:
            #     data = pickle.load(f0)
            # with open("./dataset/hawkes/data_val_hawkes_related_20.pkl", 'rb') as f0:
            #     data_related = pickle.load(f0)
            # # 时间标签-------------
            with open("./dataset/weibo_tgt_valid_time.pkl", 'rb') as f0:
                data_time = pickle.load(f0)

            # # twitter -------------
            # with open("./dataset/twitter/Twitter_val_20_to_Frets.pkl", 'rb') as f0:
            #     twitter_data = pickle.load(f0)
            # # weibo22---------
            # with open("./dataset/weibo22/weibo22_val_20_to_Frets.pkl", 'rb') as f0:
            #     weibo22_data = pickle.load(f0)
            with open("./dataset/weibo21/weibo21_val_20_to_Frets.pkl", 'rb') as f0:
                weibo21_data = pickle.load(f0)

            # Base——————————————
            # with open("./dataset/Base_data/21/21_val_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_val_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_val_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_val_manhattan_20.pkl", 'rb') as f0:
            #     weibo21_data = pickle.load(f0)
            # 22Base——————————————
            # with open("./dataset/Base_data/22/22_val_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_val_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_val_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_val_manhattan_20.pkl", 'rb') as f0:
            #     weibo22_data = pickle.load(f0)
            # twitter Base——————————————
            # with open("./dataset/Base_data/twitter/twitter_val_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_val_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_val_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_val_manhattan_20.pkl", 'rb') as f0:
            #     twitter_data = pickle.load(f0)

            # 16 Base——————————————
            # with open("./dataset/Base_data/16/16_val_cosine_20.pkl", 'rb') as f0:
                # with open("./dataset/Base_data/16/16_val_euclidean_20.pkl", 'rb') as f0:
                # with open("./dataset/Base_data/16/16_val_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/16/16_val_manhattan_20.pkl", 'rb') as f0:
            #     weibo16_data = pickle.load(f0)

        if self.set_type == 2:
            # 旧的
            # with open("./dataset/haoyu/weibo_test.pkl", 'rb') as f0:
            # with open("./dataset/weibo_tgt_test_1.pkl", 'rb') as f0:

            # 16-----------------
            # with open("./dataset/hawkes/data_test_hawkes.pkl", 'rb') as f0:
            #     data = pickle.load(f0)
            # with open("./dataset/hawkes/data_test_hawkes_related_20.pkl", 'rb') as f0:
            #     data_related = pickle.load(f0)
            # # 时间标签-------------
            with open("./dataset/weibo_tgt_test_time.pkl", 'rb') as f0:
                data_time = pickle.load(f0)

            # # twitter -------------
            # with open("./dataset/twitter/Twitter_test_20_to_Frets.pkl", 'rb') as f0:
            #     twitter_data = pickle.load(f0)
            # # weibo22---------
            # with open("./dataset/weibo22/weibo22_test_20_to_Frets.pkl", 'rb') as f0:
            #     weibo22_data = pickle.load(f0)
            with open("./dataset/weibo21/weibo21_test_20_to_Frets.pkl", 'rb') as f0:
                weibo21_data = pickle.load(f0)

            # 21Base——————————————
            # with open("./dataset/Base_data/21/21_test_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_test_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_test_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/21/21_test_manhattan_20.pkl", 'rb') as f0:
            #     weibo21_data = pickle.load(f0)

            # 22Base——————————————
            # with open("./dataset/Base_data/22/22_test_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_test_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_test_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/22/22_test_manhattan_20.pkl", 'rb') as f0:
            #     weibo22_data = pickle.load(f0)

            # twitter Base——————————————
            # with open("./dataset/Base_data/twitter/twitter_test_cosine_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_test_euclidean_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_test_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/twitter/twitter_test_manhattan_20.pkl", 'rb') as f0:
            #     twitter_data = pickle.load(f0)

            # 16 Base——————————————
            # with open("./dataset/Base_data/16/16_test_cosine_20.pkl", 'rb') as f0:
                # with open("./dataset/Base_data/16/16_test_euclidean_20.pkl", 'rb') as f0:
                # with open("./dataset/Base_data/16/16_test_pearson_20.pkl", 'rb') as f0:
            # with open("./dataset/Base_data/16/16_test_manhattan_20.pkl", 'rb') as f0:
            #     weibo16_data = pickle.load(f0)

        # data = np.array(new_data[0])
        # data_related = np.array(new_data[1])

        # 16年设置
        # data = np.array(data)
        # data_related = np.array(data_related[2])

        # 20年设置
        # data = np.array(new_data[0])
        # data_related = np.array(new_data[1])

        # twitter
        # data = weibo16_data[0]
        # data_related = weibo16_data[1]

        # twitter
        # data = twitter_data[0]
        # data_related = twitter_data[1]

        # weibo22
        # data = weibo22_data[0]
        # data_related = weibo22_data[1]

        # weibo21
        data = weibo21_data[0]
        data_related = weibo21_data[1]


        #选择多少长度的 去除最后一维的第10到15个数据点
        # data = np.delete(data, np.s_[15:120], axis=1)
        # data_related = np.delete(data_related, np.s_[15:120], axis=2)


        # sixteenth_value = data_related[:, :, 15:16]  # This slices the 16th value
        # # Replicate this value 105 times
        # replicated_values = np.repeat(sixteenth_value, 105, axis=2)
        # # Concatenate the original data with these replicated values
        # # First, take the initial 15 values
        # initial_values = data_related[:, :, :15]
        # # Concatenate initial values, the 16th value, and the replicated values
        # new_data_related = np.concatenate([initial_values, sixteenth_value, replicated_values], axis=2)
        # data_related = new_data_related
        # 20
        # number 确定用多长的x
        number = self.number
        data_y = data

        # total_sum = data[:, number:120].sum(axis=1)
        # 用计算得到的总和替换第121列
        # data[:, 120] = total_sum


        # 标签转成总量

        # total_data = np.cumsum(data_y[:, :120], axis=1)
        # data_y = np.concatenate((total_data, data_y[:, 120][:, None]), axis=1)

        print(data.shape)
        a = data_related[0]

        # data_related = data_related[2]
        # print(np.array(new_data[0]).shape)

        #20年插入
        # average_values = np.mean(new_data_related[:, :10, 15:120], axis=1)
        # new_data = np.zeros((data.shape[0], 121))  # 形状为 (10000, 121)
        # # 填充原始 data 的前15个点
        # new_data[:, :15] = data[:, :15]
        # # 插入计算得到的平均值
        # new_data[:, 15:120] = average_values
        # # 填充原始 data 的第16个点到最后
        # new_data[:, 120] = data[:, 15]
        # data = new_data
        # data_y = new_data
        ##20




        # 16年设置
        # 试试全部用真实值
        data = np.concatenate((data[:, :number], data[:, -1:]), axis=1)
        sums_for_insertion = data_related[:, :self.enc_number, number - 1:119].sum(axis=1)
        # Now divide by 10 to get the average including zeros and round down to the nearest integer
        averages_for_insertion = np.floor(sums_for_insertion / self.enc_number)

        # 插入105个0
        # averages_for_insertion[:] = 0

        # Step 2: 创建一个新的数组来插入平均值
        new_data = np.zeros((data.shape[0], data.shape[1] + 120 - number))
        new_data[:, :number] = data[:, :number]  # 前15个点
        new_data[:, number: -1] = averages_for_insertion  # 插入的105个平均值
        new_data[:, -1] = data[:, -1]
        # 不往里面补充节点
        data = new_data
        # # 16

        # # 4.7 验证105个零对预测的作用
        # # 创建一个新数组，形状为 (31780, 121)，所有值初始化为零
        # expanded_data = np.zeros((new_data[0].shape[0], 121))
        #
        # # 将原始数据复制到新数组的前15个位置
        # expanded_data[:, :15] = new_data[0][:, :15]
        #
        # # 将原始数据的最后一个数复制到新数组的第121个位置
        # expanded_data[:, 120] = new_data[0][:, 15]
        #
        # self.data_x = expanded_data
        # self.data_y = expanded_data
        # # 4.7

        # 保留前十五个点以及最后一个点
        # data = np.concatenate((data[:, :15], data[:, -1:]), axis=1)


        self.data_x = data
        self.data_y = data_y

        # self.data_x = new_data[0]
        # self.data_y = new_data[0]
        self.data_related = data_related
        self.data_time = data_time

    def __getitem__(self, index):

        number = self.number

        temp_data = self.data_x[index] # data_x 是list
        seq_y = self.data_y[index]
        temp_data = torch.tensor(temp_data, dtype=torch.float64, device='cuda')

        temp_data_reshaped = temp_data.view(-1, 1)
        # 有检索的数据
        temp_related = self.data_related[index]
        temp_related = torch.tensor(temp_related, dtype=torch.float64, device='cuda')
        # seq_data = temp_data[0: self.seq_len + self.pred_len]
        # seq_data = seq_data.reshape(-1, 1)

        start = 0  # 起始索引，Python索引从0开始
        end = self.enc_number  # 结束索引，包含在内
        #使用切片操作取出指定范围的数据
        temp_related_slice = temp_related[start:end, :]

        # 现在我们将两个张量在维度1上拼接
        # 因为temp_related是(11, 121)，我们需要将其转置以匹配temp_data的形状
        # temp_related_transposed = temp_related_slice.transpose(0, 1)
        # 新的，转置
        temp_related_transposed = temp_related_slice.T
        # 拼接得到形状为(121, 12)的张量，其中temp_data是最后一列

        temp_array = torch.cat((temp_related_transposed, temp_data_reshaped), dim=1)

        seq_data = temp_array

        # 只有一个序列
        # seq_data = temp_data_reshaped

        # 如果没有检索到相关性比较强的,那就补充自己
        # temp_data_reshaped = temp_data.view(-1, 1)
        # seq_data = torch.cat([temp_data_reshaped] * (end - start + 1), dim=1)
        # 如果没有检索到相关性比较强的,那就补充自己



        # min max scaler
        # mms = MinMaxScaler(feature_range=(0, 1))
        # seq_data = mms.fit_transform(seq_data)
        # mms.fit(seq_data[0:15])
        # seq_data[0:15] = mms.transform(seq_data[0:15])
        # seq_data[15:16] = mms.transform(seq_data[15:16])


        seq_x = seq_data[0:self.seq_len]

        # seq_y = np.concatenate((b, c), axis=0)

        # 16年
        seq_y = seq_y[number: ]

        # 20年\啥都没有，只有15长的
        # seq_y = np.concatenate((seq_y[self.seq_len - self.label_len:number], seq_y[120:]))

        # 20年\啥都没有，只有15长的
        # seq_y = seq_y[self.seq_len-self.label_len:]
        if index >= index :
            index_mark = index - index
        start_mark = pd.Timestamp.fromtimestamp(int(self.data_time[index_mark]))

        # 使用 start_mark 作为起点生成时间序列
        time_series = pd.date_range(start=start_mark, periods=self.seq_len + self.pred_len, freq='12T')

        # Create a DataFrame from the time series
        mark = pd.DataFrame(time_series, columns=['datetime'])

        # Extract components from datetime and drop the datetime column
        mark['month'] = mark['datetime'].dt.month
        mark['day'] = mark['datetime'].dt.day
        mark['weekday'] = mark['datetime'].dt.weekday
        mark['hour'] = mark['datetime'].dt.hour
        mark.drop(columns='datetime', inplace=True)

        # Convert DataFrame to numpy array if necessary
        seq_x_mark = mark.iloc[:self.seq_len].values
        seq_y_mark = mark.iloc[number:self.seq_len + 1].values

        seq_y = seq_y.reshape(self.pred_len, 1)

        # 只要15个
        # seq_x = seq_x[:15, :]
        # seq_y = seq_y[-7:, :]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        mms = MinMaxScaler(feature_range=(0, 1))
        return mms.fit_transform(data.cpu())
        # return self.scaler.inverse_transform(data)

