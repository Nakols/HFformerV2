import numpy as np
from scipy.signal import lfiltic, lfilter

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x
    
    def transform(self, x):
        normalized_x = (x - self.mu)/self.sd
        return normalized_x    

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu


class MinMax():
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit_transform(self, x):
        self.min = np.min(x, axis=0, keepdims=True)
        self.max = np.max(x, axis=0, keepdims=True)
        minmax_x = (x-self.min)/(self.max-self.min)
        return minmax_x

    def transform(self, x):
        minmax_x = (x-self.min)/(self.max-self.min)
        return minmax_x
    
    def inverse_transform(self, x):
        return x*(self.max-self.min)+self.min
      

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def moving_average_standardization(prices, quantities, window):
    prices = np.array(prices)
    idx = 0
    segments = (len(prices)//window)-1
    prices_deavg = np.zeros_like(prices)
    
    for segment in range(segments):
        segment_avg = np.average(prices[idx:idx+window], weights=quantities[idx:idx+window])
        segment_prices_deavg = np.subtract(prices[idx+window:idx+2*window], segment_avg)
        prices_deavg[idx:idx+window] = segment_prices_deavg
        idx += window
    
    return prices_deavg 
  
  
def log(prices):
  return np.log(prices)


def consecutive_diff(prices):
  return np.ediff1d(prices)


def consecutive_window_diff(prices, window=1):
  prices = np.array(prices)
  p1 = prices[window:]
  p2 = prices[:-window]
  return p1-p2


def ewma_linear_filter(array, window):
    alpha = 2 /(window + 1)
    b = [alpha]
    a = [1, alpha-1]
    zi = lfiltic(b, a, array[0:1], [0])
    return lfilter(b, a, array, zi=zi)[0]