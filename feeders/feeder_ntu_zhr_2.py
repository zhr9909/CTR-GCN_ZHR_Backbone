import numpy as np

from torch.utils.data import Dataset

from feeders import tools_zhr


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)

        self.data_1 = npz_data['camera1_x']
        self.label_1 = np.where(npz_data['camera1_y'] > 0)[1]
        self.sample_name_1 = ['train_' + str(i) for i in range(len(self.data_1))]


        self.data_2 = npz_data['camera2_x']
        self.label_2 = np.where(npz_data['camera2_y'] > 0)[1]
        self.sample_name_2 = ['test_' + str(i) for i in range(len(self.data_2))]

        N, T, _ = self.data_1.shape
        self.data_1 = self.data_1.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.data_2 = self.data_2.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label_1)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # data_1的处理
        data_numpy_1 = self.data_1[index]
        label_1 = self.label_1[index]
        data_numpy_1 = np.array(data_numpy_1)
        valid_frame_num_1 = np.sum(data_numpy_1.sum(0).sum(-1).sum(-1) != 0)

        # reshape Tx(MVC) to CTVM
        data_numpy_1 = tools_zhr.valid_crop_resize(data_numpy_1, valid_frame_num_1, self.p_interval, self.window_size)
        # print('返回的data：',data_numpy_1.shape)
        if self.random_rot:
            data_numpy_1 = tools_zhr.random_rot(data_numpy_1)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy_1)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy_1[:, :, v1 - 1] - data_numpy_1[:, :, v2 - 1]
            data_numpy_1 = bone_data_numpy
        if self.vel:
            data_numpy_1[:, :-1] = data_numpy_1[:, 1:] - data_numpy_1[:, :-1]
            data_numpy_1[:, -1] = 0

        # data_2的处理
        data_numpy_2 = self.data_2[index]
        label_2 = self.label_2[index]
        data_numpy_2 = np.array(data_numpy_2)
        valid_frame_num = np.sum(data_numpy_2.sum(0).sum(-1).sum(-1) != 0)

        # reshape Tx(MVC) to CTVM
        data_numpy_2 = tools_zhr.valid_crop_resize(data_numpy_2, valid_frame_num, self.p_interval, self.window_size)
        # print('返回的data：',data_numpy_2.shape)
        if self.random_rot:
            data_numpy_2 = tools_zhr.random_rot(data_numpy_2)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy_2)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy_2[:, :, v1 - 1] - data_numpy_2[:, :, v2 - 1]
            data_numpy_2 = bone_data_numpy
        if self.vel:
            data_numpy_2[:, :-1] = data_numpy_2[:, 1:] - data_numpy_2[:, :-1]
            data_numpy_2[:, -1] = 0

        # print('最终返回的data：',data_numpy.shape)
        return data_numpy_1, label_1, index, data_numpy_2, label_2, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
