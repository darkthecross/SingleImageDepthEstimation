import numpy as np
import cv2
import sys
import glob

def calc_rmse(origin, predicted, mask):
    of = origin.flatten()
    pf = predicted.flatten()
    mf = mask.flatten()
    dd = of - pf
    ddsq = np.square(dd)
    ddsq_m = np.multiply(ddsq, mf)
    count = np.sum(mf)
    ddsum = np.sum(ddsq_m)
    rmse = np.sqrt(ddsum/count)
    return rmse

def calc_t(origin, predicted, mask, t_val):
    of = origin.flatten()
    pf = predicted.flatten()
    mf = mask.flatten()
    of += 1e-8
    pf += 1e-8
    tmp_res = np.divide(of, pf)
    tmp_res[np.where(tmp_res<1)] = 1/tmp_res[np.where(tmp_res<1)]
    tmp_res = np.multiply(tmp_res, mf)
    tmp_res[np.where(tmp_res<t_val)] = 0
    tmp_res[np.where(tmp_res>=t_val)] = 1
    return 1 - np.sum(tmp_res) / np.sum(mf)

def abs_rel_diff(origin, predicted, mask):
    of = origin.flatten()
    pf = predicted.flatten()
    mf = mask.flatten()
    dd = np.absolute(of - pf)
    dd_m = np.multiply(dd, mf)
    of += 1e-8
    res = dd_m / of
    result = np.sum(res)/np.sum(mf)
    return result

def calc_mask(origin):
    ma = np.zeros_like(origin)
    ma[np.where(origin>0.98)] = 1
    ma[np.where(origin<0.02)] = 1
    ma = 1-ma
    return ma

def main(s):
    if len(s) == 0:
        print('Error: please specify model.')
    else:
        imgpath = 'result_images/' + s[1] + '/'
        #print(imgpath)
        origin_files = glob.glob(imgpath + '*origin_dps*')
        #print(origin_files)
        total = 0
        count = 0
        for fn in origin_files:
            origin = cv2.imread(fn)
            origin_gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
            origin_gray = origin_gray / 255
            mask = calc_mask(origin_gray)
            pred_fn = fn.replace('origin', 'predicted')
            pred = cv2.imread(pred_fn)
            pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            pred_gray = cv2.resize(pred_gray, (320, 240))
            pred_gray = pred_gray / 255
            if s[2] == 'rmse':
                total += calc_rmse(origin_gray, pred_gray, mask)
            elif s[2] == 't':
                total += calc_t(origin_gray, pred_gray, mask, 1.25)
            elif s[2] == 'ard':
                total += abs_rel_diff(origin_gray, pred_gray, mask)
            count += 1
        print(total/count)

if __name__ == '__main__':
    main(sys.argv)
