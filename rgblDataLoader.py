import glob
import cv2
import numpy as np
import math
from demo import rot, get_bbox, retrive_bbox3d

class rgblDataLoader():
    def __init__(self):
        self.dirPre = ['../ROB599Perception/deploy/trainval/a*/*_image.jpg',
        '../ROB599Perception/deploy/trainval/b*/*_image.jpg',
        '../ROB599Perception/deploy/trainval/c*/*_image.jpg',
        '../ROB599Perception/deploy/trainval/d*/*_image.jpg',
        '../ROB599Perception/deploy/trainval/e*/*_image.jpg',
        '../ROB599Perception/deploy/trainval/f*/*_image.jpg']

        return None

    def loadImageNames(self):
        self.ppmFileNames = []
        for fp in self.dirPre:
            self.ppmFileNames = self.ppmFileNames + glob.glob(fp)
        test_count = 20
        #test_ids = np.flip(np.sort(np.random.choice(len(self.ppmFileNames), test_count)), 0).tolist()
        #print(test_ids)
        test_ids = [2651, 2623, 2621, 2444, 2414, 2112, 2064, 1981, 1711, 1678, 1663, 1450, 1396, 950, 661, 625, 402, 289, 157, 61]
        self.testRGBNames = []
        for test_id in test_ids:
            self.testRGBNames.append(self.ppmFileNames[test_id])
            del self.ppmFileNames[test_id]

    def getNextBatchTraining(self, batch_size):
        img = np.zeros((batch_size, 240, 320, 3), dtype=np.float32)
        dps1 = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        mask = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        data_count = len(self.ppmFileNames)
        ids = np.random.choice(data_count, batch_size)
        for i in range(batch_size):
            tmpim = cv2.imread(self.ppmFileNames[ids[i]]).astype(np.float32)
            tmpim = tmpim / 255
            img[i, :, :, :] = cv2.resize(tmpim, (320, 240))
            td = cv2.imread(self.ppmFileNames[ids[i]].replace('_image.jpg', '_dps.jpg')).astype(np.float32)
            td = cv2.cvtColor(td, cv2.COLOR_BGR2GRAY)
            td = td/255
            dps1[i, :, :, 0] = td
            td[np.where(td==1)] = 0
            td[np.where(td>0)] = 1
            mask[i, :, :, 0] = td
        return img, dps1, mask

    def getNextBatchTesting(self, batch_size):
        img = np.zeros((batch_size, 240, 320, 3), dtype=np.float32)
        dps1 = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        mask = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        data_count = len(self.testRGBNames)
        ids = np.random.choice(data_count, batch_size)
        for i in range(batch_size):
            tmpim = cv2.imread(self.testRGBNames[ids[i]]).astype(np.float32)
            tmpim = tmpim / 255
            img[i, :, :, :] = cv2.resize(tmpim, (320, 240))
            td = cv2.imread(self.ppmFileNames[ids[i]].replace('_image.jpg', '_dps.jpg')).astype(np.float32)
            td = cv2.cvtColor(td, cv2.COLOR_BGR2GRAY)
            td = td/255
            dps1[i, :, :, 0] = td
            td[np.where(td==1)] = 0
            td[np.where(td>0)] = 1
            mask[i, :, :, 0] = td
        return img, dps1, mask

    def preproc(self):
        for i in range(len(self.ppmFileNames)):
            print(i)
            tmpim = cv2.imread(self.ppmFileNames[i]).astype(np.float32)
            tmpim = tmpim / 255
            xyz = np.fromfile(self.ppmFileNames[i].replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
            xyz.resize([3, xyz.size // 3])
            # get projection matrix
            proj = np.fromfile(self.ppmFileNames[i].replace('_image.jpg', '_proj.bin'), dtype=np.float32)
            proj.resize([3, 4])
            # project clound points onto image
            uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
            uv = uv / uv[2, :]
            clr = np.linalg.norm(xyz, axis=0)
            dps = np.zeros((tmpim.shape[0], tmpim.shape[1]), dtype=np.float32)
            dpsmk = np.zeros((tmpim.shape[0], tmpim.shape[1]), dtype=np.float32)
            # https://en.wikipedia.org/wiki/Bilateral_filter
            for uvidx in range(uv.shape[1]):
                if int(uv[0, uvidx]) >= 0 and int(uv[0, uvidx]) <= dps.shape[1] and int(uv[1, uvidx]) >= 0 and int(uv[1, uvidx]) <= dps.shape[0]:
                    dps[ int(uv[1, uvidx]), int(uv[0, uvidx]) ] = clr[uvidx]
                    dpsmk[ int(uv[1, uvidx]), int(uv[0, uvidx]) ] = 1
            ksize = 11
            dps_blurred = cv2.GaussianBlur(dps, (ksize, ksize), 0)
            weight_count = cv2.GaussianBlur(dpsmk, (ksize, ksize), 0)
            weight_count += 1e-9
            res = np.divide(dps_blurred, weight_count)
            res_blurred = cv2.GaussianBlur(res, (ksize, ksize), 0)
            rm = np.amax(res_blurred)
            res_blurred = 1-res_blurred/rm
            td = cv2.resize(res_blurred, (80, 60))*255
            cv2.imwrite(self.ppmFileNames[i].replace('_image.jpg', '_dps.jpg'), td)
