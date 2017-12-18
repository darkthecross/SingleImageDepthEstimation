import h5py
import numpy as np
import cv2

class dbReader:

    def __init__(self):
        self.FILE_PATH = 'dataset/nyu_depth_v2_labeled.mat'
        self.SAVE_PATH = 'dataset/'

    def loadData(self, c, d):
        """
        'rgb' or 'grayscale'(default)
        'origin' or 'normalized'(default)
        """
        if c == 'rgb':
            fc = open(self.SAVE_PATH+'img_color.npy', 'rb')
            self.img = np.load(fc)
            fc.close()
        else:
            fimg = open(self.SAVE_PATH+'img.npy', 'rb')
            self.img = np.load(fimg)
            fimg.close()
        if d == 'origin':
            fdps = open(self.SAVE_PATH+'dps_origin.npy', 'rb')
            self.depth = np.load(fdps)
            fdps.close()
        else:
            fdps = open(self.SAVE_PATH+'dps.npy', 'rb')
            self.depth = np.load(fdps)
            fdps.close()
        rng_state = np.random.get_state()
        np.random.shuffle(self.img)
        np.random.set_state(rng_state)
        np.random.shuffle(self.depth)
        sz = self.img.shape[0]
        """
        for i in range(0, sz):
            average_color = [self.img[i, :, :, k].mean() for k in range(self.img.shape[-1])]
            self.img[i, :, :] -= average_color
        """

    def getNextBatchResized(self, batch_size, ratio):
        """
        make sure batch_size <= data_count
        """
        data_count = self.img.shape[0]
        channel_count = len(self.img.shape)
        ids = np.random.choice(data_count, batch_size)
        im = np.zeros((batch_size, self.img.shape[1], self.img.shape[2], int(channel_count*2-5)), dtype=np.float32)
        dp = np.zeros((batch_size, int(self.depth.shape[1]/ratio), int(self.depth.shape[2]/ratio), 1), dtype=np.float32)
        for i in range(batch_size):
            if channel_count == 3:
                im[i, :, :, 0] = self.img[ids[i], :, :]
            else:
                im[i, :, :, :] = self.img[ids[i], :, :, :]
            dp[i, :, :, 0] = cv2.resize(self.depth[ids[i], :, :],None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
        return [im, dp]

    def getNextBatchResizedTraining(self, batch_size, ratio):
        """
        make sure batch_size <= data_count
        """
        data_count = 1399
        channel_count = len(self.img.shape)
        ids = np.random.choice(data_count, batch_size)
        im = np.zeros((batch_size, self.img.shape[1], self.img.shape[2], int(channel_count*2-5)), dtype=np.float32)
        dp = np.zeros((batch_size, int(self.depth.shape[1]/ratio), int(self.depth.shape[2]/ratio), 1), dtype=np.float32)
        flip = np.random.rand()>0.5
        for i in range(batch_size):
            if channel_count == 3:
                if flip:
                    im[i, :, :, 0] = np.flip(self.img[ids[i], :, :], 1)
                else:
                    im[i, :, :, 0] = self.img[ids[i], :, :]
            else:
                if flip:
                    im[i, :, :, :] = np.flip(self.img[ids[i], :, :, :], 1)
                else:
                    im[i, :, :, :] = self.img[ids[i], :, :, :]
            dp[i, :, :, 0] = cv2.resize(self.depth[ids[i], :, :],None,fx=1/ratio, fy=1/ratio, interpolation = cv2.INTER_CUBIC)
            if flip:
                dp[i, :, :, 0] = np.flip(dp[i, :, :, 0], 1)
        return [im, dp]

    def dataAugmentation(self, idx, ratio):
        da_type = np.random.rand()
        channel_count = len(self.img.shape)
        nim = np.zeros((self.img.shape[1], self.img.shape[2], int(channel_count*2-5)), dtype=np.float32)
        ndp = np.zeros((int(self.depth.shape[1]/ratio), int(self.depth.shape[2]/ratio), 1), dtype=np.float32)
        imgw = self.img.shape[1]
        imgh = self.img.shape[2]
        if da_type < 0.5:
            # crop
            sub_id = np.random.rand()
            x_id = sub_id<0.5
            x_l = int (int(x_id) * float(imgw) / 2)
            x_h = x_l + int( float(imgw) / 2 )
            y_id = sub_id<0.25 or sub_id > 0.75
            y_l = int (int(y_id) * float(imgh) / 2)
            y_h = y_l + int( float(imgh) / 2 )
            if channel_count == 3:
                tmpim = self.img[idx, x_l:x_h, y_l:y_h, :]
            else:
                tmpim = self.img[idx, x_l:x_h, y_l:y_h]
            nim = cv2.resize(tmpim, None, fx=ratio/2, fy=ratio/2, interpolation = cv2.INTER_CUBIC)
            ndp[:, :, 0] = cv2.resize(self.depth[idx, x_l:x_h, y_l:y_h], None, fx=2/ratio, fy=2/ratio, interpolation = cv2.INTER_CUBIC)
        else:
            if channel_count == 3:
                nim = self.img[idx, :, :, :]
            else:
                nim = self.img[idx, :, :]
            ndp[:, :, 0] = cv2.resize(self.depth[idx, :, :], None, fx=1/ratio, fy=1/ratio, interpolation = cv2.INTER_CUBIC)
        if da_type < 0.25 or da_type > 0.75:
            # flip
            if channel_count == 3:
                nim[:, :, 0] = np.flip(nim[:, :, 0], 1)
            else:
                nim[:, :, :] = np.flip(nim[:, :, :], 1)
            ndp[:, :, 0] = np.flip(ndp[:, :, 0], 1)
        maxi = np.amax(ndp)
        ndp[:, :, 0] = ndp[:, :, 0]/maxi
        return [nim, ndp]


    def dataAugmentationCrop(self, idx, ratio):
        da_type = np.random.rand()
        channel_count = len(self.img.shape)
        nim = np.zeros((self.img.shape[1], self.img.shape[2], int(channel_count*2-5)), dtype=np.float32)
        ndp = np.zeros((int(self.depth.shape[1]/ratio), int(self.depth.shape[2]/ratio), 1), dtype=np.float32)
        imgw = self.img.shape[1]
        imgh = self.img.shape[2]
        if da_type < 0.5:
            # crop
            x_l = np.random.randint(imgw/2)
            x_h = np.random.randint(x_l + imgw/2, imgw)
            y_l = np.random.randint(imgh/2)
            y_h = int( float(imgh) * float(imgw) * float(x_h - x_l) )
            if channel_count == 3:
                tmpim = self.img[idx, x_l:x_h, y_l:y_h, :]
            else:
                tmpim = self.img[idx, x_l:x_h, y_l:y_h]
            nim = cv2.resize(tmpim, (int(imgh), int(imgw)), interpolation = cv2.INTER_CUBIC)
            ndp[:, :, 0] = cv2.resize(self.depth[idx, x_l:x_h, y_l:y_h], (int(imgh/ratio), int(imgw/ratio)), interpolation = cv2.INTER_CUBIC)
        else:
            if channel_count == 3:
                nim = self.img[idx, :, :, :]
            else:
                nim = self.img[idx, :, :]
            ndp[:, :, 0] = cv2.resize(self.depth[idx, :, :], None, fx=1/ratio, fy=1/ratio, interpolation = cv2.INTER_CUBIC)
        if da_type < 0.25 or da_type > 0.75:
            # flip
            if channel_count == 3:
                nim[:, :, 0] = np.flip(nim[:, :, 0], 1)
            else:
                nim[:, :, :] = np.flip(nim[:, :, :], 1)
            ndp[:, :, 0] = np.flip(ndp[:, :, 0], 1)
        maxi = np.amax(ndp)
        mini = np.amin(ndp)
        ndp[:, :, 0] = (ndp[:, :, 0]-mini)/(maxi-mini)
        return [nim, ndp]

    def getNextBatchResizedTrainingNew(self, batch_size, ratio):
        """
        make sure batch_size <= data_count
        """
        data_count = 1399
        channel_count = len(self.img.shape)
        ids = np.random.choice(data_count, batch_size)
        im = np.zeros((batch_size, self.img.shape[1], self.img.shape[2], int(channel_count*2-5)), dtype=np.float32)
        dp = np.zeros((batch_size, int(self.depth.shape[1]/ratio), int(self.depth.shape[2]/ratio), 1), dtype=np.float32)
        flip = np.random.rand()>0.5
        for i in range(batch_size):
            if channel_count == 3:
                [im[i, :, :, 0], dp[i, :, :, :]] = self.dataAugmentation(ids[i], ratio)
            else:
                [im[i, :, :, :], dp[i, :, :, :]] = self.dataAugmentation(ids[i], ratio)
        return [im, dp]

    def getNextBatchResizedTrainingWithRandCrop(self, batch_size, ratio):
        """
        make sure batch_size <= data_count
        """
        data_count = 1399
        channel_count = len(self.img.shape)
        ids = np.random.choice(data_count, batch_size)
        im = np.zeros((batch_size, self.img.shape[1], self.img.shape[2], int(channel_count*2-5)), dtype=np.float32)
        dp = np.zeros((batch_size, int(self.depth.shape[1]/ratio), int(self.depth.shape[2]/ratio), 1), dtype=np.float32)
        flip = np.random.rand()>0.5
        for i in range(batch_size):
            if channel_count == 3:
                [im[i, :, :, 0], dp[i, :, :, :]] = self.dataAugmentationCrop(ids[i], ratio)
            else:
                [im[i, :, :, :], dp[i, :, :, :]] = self.dataAugmentationCrop(ids[i], ratio)
        return [im, dp]

    def getTest(self, ratio, sz=50):
        """
        make sure batch_size <= data_count
        """
        channel_count = len(self.img.shape)
        im = np.zeros((sz, self.img.shape[1], self.img.shape[2], int(channel_count*2-5)), dtype=np.float32)
        dp = np.zeros((sz, int(self.depth.shape[1]/ratio), int(self.depth.shape[2]/ratio), 1), dtype=np.float32)
        for i in range(1399,1399+sz):
            if channel_count == 3:
                im[i-1399, :, :, 0] = self.img[i, :, :]
            else:
                im[i-1399, :, :, :] = self.img[i, :, :, :]
            dp[i-1399, :, :, 0] = cv2.resize(self.depth[i, :, :],None,fx=1/ratio, fy=1/ratio, interpolation = cv2.INTER_CUBIC)
        return [im, dp]
