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

        number = self.number
        data_y = data

        # total_sum = data[:, number:120].sum(axis=1)
        # 用计算得到的总和替换第121列
        # data[:, 120] = total_sum

        print(data.shape)
        a = data_related[0]

        data = np.concatenate((data[:, :number], data[:, -1:]), axis=1)
        sums_for_insertion = data_related[:, :self.enc_number, number - 1:119].sum(axis=1)
        # Now divide by 10 to get the average including zeros and round down to the nearest integer
        averages_for_insertion = np.floor(sums_for_insertion / self.enc_number)
        new_data = np.zeros((data.shape[0], data.shape[1] + 120 - number))
        new_data[:, :number] = data[:, :number]  # 前15个点
        new_data[:, number: -1] = averages_for_insertion  # 插入的105个平均值
        new_data[:, -1] = data[:, -1]
        # 不往里面补充节点
        data = new_data

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
        temp_related_transposed = temp_related_slice.T
        temp_array = torch.cat((temp_related_transposed, temp_data_reshaped), dim=1)
        seq_data = temp_array


        seq_x = seq_data[0:self.seq_len]
        seq_y = seq_y[number: ]


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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        mms = MinMaxScaler(feature_range=(0, 1))
        return mms.fit_transform(data.cpu())
        # return self.scaler.inverse_transform(data)
