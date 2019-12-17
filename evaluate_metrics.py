import os
import click
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def sq_sinv(y,y_):
    # To avoid log(0) = -inf
    y_[y_==0] = 1
    y[y==0] = 1
    alpha = np.mean(np.log(y_) - np.log(y))
    err = (np.log(y) - np.log(y_) + alpha) ** 2
    return (np.mean(err[:]) / 2)

def pear_coeff(y,y_):
    y = y.ravel()
    y_ = y_.ravel()
    err = pearsonr(y,y_)
    return err[0]

@click.command()
@click.option('--gt_path',
              type=click.STRING,
              default='',
              help='Path to the folder containing the ground-truth images')
@click.option('--results_path',
              type=click.STRING,
              default='',
              help='Path to the folder containing the ground-truth images')

def calculate_metrics(gt_path, results_path):
    l1 = os.listdir(gt_path)
    l1.sort()
    l2 = os.listdir(results_path)
    l2.sort()

    score = []
    names = []
    for i in range(len(l1)):
        g1 = (io.imread(os.path.join(gt_path,l1[i]))/ 256).astype(np.uint8)
        t1 = io.imread(os.path.join(results_path,l2[i])) 
        # If depth value is nan or invalid, change it to zero.
        for j in range(g1.shape[0]):
            for k in range(g1.shape[1]):
                if g1[j,k] >= 255:
                    g1[j,k] = 0
                    t1[j,k] = 0

        score.append([sq_sinv(g1,t1),pear_coeff(g1,t1)])
        print('RMSE Squared log Scale-invariant error:', score[-1][0], ', Pearson Coefficient', score[-1][1])
        names.append(l1[i])

    m = np.mean(np.array(score), axis = 0)
    print('Mean RMSE Squared log Scale-invariant error', m[0])
    print('Mean Pearson Coefficient', m[1])

if __name__=='__main__':
    calculate_metrics()