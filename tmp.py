from rgblDataLoader import rgblDataLoader
import cv2

rd = rgblDataLoader()

rd.loadImageNames()


img, dps, msk = rd.getNextBatchTesting(10)

for i in range(10):
    cv2.imshow('im', img[i, :, :, :])
    cv2.waitKey(0)
    cv2.imshow('dp', dps[i, :, :, :])
    cv2.waitKey(0)
    cv2.imshow('msk', msk[i, :, :, :])
    cv2.waitKey(0)
