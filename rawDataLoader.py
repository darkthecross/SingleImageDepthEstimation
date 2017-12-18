import glob
import cv2
import numpy as np

class rawDataLoader():
    def __init__(self):
        self.dirPre = ['dataset/reception_rooms/reception_room_0001a/',
        'dataset/reception_rooms/reception_room_0001b/',
        'dataset/reception_rooms/reception_room_0003/',
        'dataset/reception_rooms/reception_room_0004/']
        return None

    def loadImageNames(self):
        self.ppmFileNames = []
        self.pgmFileNames = []
        for roomid in range(4):
            idxFile = open(self.dirPre[roomid] + 'INDEX.txt')
            allTexts = idxFile.read()
            allTextArray = allTexts.split('\n')
            i = 0
            while i < len(allTextArray):
                if '.ppm' in allTextArray[i]:
                    self.ppmFileNames.append(self.dirPre[roomid] + allTextArray[i])
                    while '.pgm' not in allTextArray[i]:
                        i += 1
                    self.pgmFileNames.append(self.dirPre[roomid] + allTextArray[i])
                    i += 1
                else:
                    i += 1
        test_count = 20
        #test_ids = np.flip(np.sort(np.random.choice(len(self.ppmFileNames), test_count)), 0).tolist()
        #print(test_ids)
        test_ids = [2675, 2652, 2621, 2444, 2414, 2112, 2064, 1981, 1711, 1678, 1663, 1450, 1396, 950, 661, 625, 402, 289, 157, 61]
        self.testRGBNames = []
        self.testDepthNames = []
        for test_id in test_ids:
            self.testRGBNames.append(self.ppmFileNames[test_id])
            self.testDepthNames.append(self.pgmFileNames[test_id])
            del self.ppmFileNames[test_id]
            del self.pgmFileNames[test_id]

    def getNextBatchTraining(self, batch_size):
        img = np.zeros((batch_size, 240, 320, 3), dtype=np.float32)
        dps = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        mask = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        data_count = len(self.ppmFileNames)
        ids = np.random.choice(data_count, batch_size)
        for i in range(batch_size):
            tmpim = cv2.imread(self.ppmFileNames[ids[i]]).astype(np.float32)
            tmpim = tmpim / 255
            img[i, :, :, :] = cv2.resize(tmpim, (320, 240))
            tmpdp = cv2.imread(self.pgmFileNames[ids[i]]).astype(np.float32)
            tmpdp = tmpdp / 255
            tmpdp = cv2.cvtColor(tmpdp, cv2.COLOR_BGR2GRAY)
            td = cv2.resize(tmpdp, (80, 60))
            dps[i, :, :, 0] = td
            td[np.where(td==1)] = 0
            td[np.where(td>0)] = 1
            mask[i, :, :, 0] = td
        return img, dps, mask

    def getNextBatchTesting(self, batch_size):
        img = np.zeros((batch_size, 240, 320, 3), dtype=np.float32)
        dps = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        mask = np.zeros((batch_size, 60, 80, 1), dtype=np.float32)
        data_count = len(self.testRGBNames)
        ids = np.random.choice(data_count, batch_size)
        for i in range(batch_size):
            tmpim = cv2.imread(self.testRGBNames[ids[i]]).astype(np.float32)
            tmpim = tmpim / 255
            img[i, :, :, :] = cv2.resize(tmpim, (320, 240))
            tmpdp = cv2.imread(self.testDepthNames[ids[i]]).astype(np.float32)
            tmpdp = tmpdp / 255
            tmpdp = cv2.cvtColor(tmpdp, cv2.COLOR_BGR2GRAY)
            td = cv2.resize(tmpdp, (80, 60))
            dps[i, :, :, 0] = td
            td[np.where(td==1)] = 0
            td[np.where(td>0)] = 1
            mask[i, :, :, 0] = td
        return img, dps, mask
