import glob
import cv2

files = glob.glob('result_images/Eigen_pretrain/*pred*')
for fn in files:
    img = cv2.imread(fn)
    imgre = cv2.resize(img, (80, 60))
    imgre = cv2.medianBlur(imgre,2)
    img = cv2.resize(imgre, (320, 240))
    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.imwrite( fn.replace('predicted', 'processed'), img)
