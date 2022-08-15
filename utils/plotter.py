import numpy as np
import matplotlib.pyplot as plt

def timediff(arr, plot=True):
    difftime = (arr[1:] - arr[:-1]).astype('int')
    if plot:
        plt.hist(difftime, bins=[0,5,10,15,20,30,40,50,100,150,200,250,300,350,400,500,600,700,800])
    return difftime, np.min(difftime), np.mean(difftime), np.max(difftime)

    
    

 