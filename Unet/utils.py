# -*- coding: utf-8 -*-
import numpy as np
import torch


def myprint(record_file, *args):
    """Print & Record while training."""
    print(*args)
    f = open(record_file, 'a')
    print(*args, file=f)
    f.close()


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█',content =None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if content:
        print('\r%s |%s| %s%% %s %s' % (prefix, bar, percent, suffix, content), end = ' ')
    else:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = ' ')

    # Print New Line on Complete
    if iteration == total:
        print()


def getIOU(SR, GT):
    """
    都是二值图
    :param SR: binary image
    :param GT: binary image
    :return:
    """
    TP = (SR + GT == 2).astype(np.float32)
    FP = (SR + (1 - GT) == 2).astype(np.float32)
    FN = ((1 - SR) + GT == 2).astype(np.float32)

    IOU = float(np.sum(TP)) / (float(np.sum(TP + FP + FN)) + 1e-6)

    return IOU


def getDSC(SR, GT):
    """
    都是二值图
    :param SR: binary image
    :param GT: binary image
    :return:
    """

    Inter = np.sum(((SR + GT) == 2).astype(np.float32))
    DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

    return DC


