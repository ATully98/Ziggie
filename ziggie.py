import numpy as np
import scipy as sp
from scipy.signal import detrend
import matplotlib as mpl
from matplotlib import pyplot as plt

def get_ziggie(Xpts, Ypts, seg_min, seg_max):
    Xn = Xpts[seg_min:seg_max+1]
    Xn = Xn/Xn[-1]
    Yn = Ypts[seg_min:seg_max+1]
    Yn = Yn/Yn[0]
    n = len(Xn)
    # find best fit line
    U = detrend(Xn, type = "constant", axis = 0) # (Xi-Xbar)
    V = detrend(Yn, type = "constant", axis = 0) # (Yi-Ybar)
    b = np.sign(np.sum(U*V))*np.std(Yn, ddof = 1)/np.std(Xn, ddof = 1);
    Y_int = np.mean(Yn) - b*np.mean(Xn)
    X_int = -Y_int/b

    # Project the data onto the best-fit line
    Rev_x = (Yn - Y_int) / b # The points reflected about the bes-fit line
    Rev_y = b * Xn + Y_int
    x_prime = (Xn + Rev_x)/2   # Average the both sets to get the projected points
    y_prime = (Yn + Rev_y)/2

    # Get the TRM, NRM, and line lengths
    Delta_x_prime = np.abs( np.amax(x_prime)-np.amin(x_prime) )
    Delta_y_prime = np.abs( np.amax(y_prime)-np.amin(y_prime) )
    Line_Len = np.sqrt(Delta_x_prime**2 + Delta_y_prime**2)

    # Set cumulative length to 0
    cum_len = 0.0

    # iterate through pairs of points in Arai plot
    for i in range(0, n-1):

        # find the distance between the two points
        dist = np.sqrt((Xn[i+1,0] - Xn[i,0])**2 + (Yn[i+1,0] - Yn[i,0])**2)
        # Add to the cumulative distance
        cum_len = cum_len + dist

    # calculate the log of the cumulative length over the length of the best fit line
    ziggie = np.log(cum_len/Line_Len)
    return ziggie


def example():

  # Input x-points
  Xpts = np.array(
  [[0.        ],
   [0.10279781],
   [0.08338857],
   [0.0771133 ],
   [0.20523519],
   [0.21142007],
   [0.31138025],
   [0.39184652],
   [0.41653954],
   [0.53064473],
   [0.52430405],
   [0.55089284],
   [0.56472051],
   [0.5898137 ]])
  
  # Input y-points
  Ypts = np.array(
  [[2.36163382e+00],
   [2.26145805e+00],
   [2.05552208e+00],
   [1.88568978e+00],
   [1.59950356e+00],
   [1.36308176e+00],
   [9.35608325e-01],
   [6.97214393e-01],
   [4.14168706e-01],
   [2.06242754e-01],
   [1.02009822e-01],
   [7.57239946e-02],
   [2.34832109e-02],
   [2.21944413e-03]])
  
  # Segment of Arai plot to be used (0 to n)
  seg_min = 4
  seg_max = 9
  seg = np.arange(seg_min, seg_max+1,1)
  
  #number of points
  n = seg_max-seg_min+1
  
  
  b, Y_int = get_gradients(Xpts, Ypts, seg)
  ziggie = get_ziggie(Xpts, Ypts, seg_min, seg_max)
  return ziggie

