import numpy as np

def sdf2circle(nrow,ncol, ic,jc,r):
    
    x = np.linspace(1, ncol, ncol)
    y = np.linspace(1, nrow, nrow)
    X, Y = np.meshgrid(x,y)
    sdf = np.sqrt((X-ic)**2+(Y-jc)**2)-r;
    
    return sdf