# -*- coding: utf-8 -*-
"""

   This function updates the level set function according to the CV model 
   input: 
       I: input image
       phi0: level set function to be updated
       mu: weight for length term
       nu: weight for area term, default value 0
       lambda_1:  weight for c1 fitting term
       lambda_2:  weight for c2 fitting term
       muP: weight for level set regularization term 
       timestep: time step
       epsilon: parameter for computing smooth Heaviside and dirac function
       numIter: number of iterations
   output: 
       phi: updated level set function

"""
import numpy as np


def evol_cv(I, phi0, nu, lambda_1, lambda_2, timestep, epsilon):
    phi = phi0

    phi = NeumannBoundCond(phi)
    diracPhi=Delta_h(phi,epsilon)
    Hphi=Heaviside(phi, epsilon)
    [C1,C2]=binaryfit(I,Hphi)
        
    phi = phi+timestep*(diracPhi*(nu-lambda_1*((I-C1)**2)+lambda_2*((I-C2)**2)))
    
    return phi

def Heaviside(phi, epsilon):
    H = 0.5*(1+(2/np.pi))*np.arctan(phi/epsilon)
    return H

def Delta_h(phi, epsilon):
    D = (epsilon/np.pi)/(epsilon**2+phi**2)
    return D

def NeumannBoundCond(x=[]):
    
    g = x
    g[0,0] = g[2,2]
    g[0,-1] = g[2,-3]
    g[-1, 0] = g[-3, 2]
    g[-1,-1] = g[-3,-3]
    g[0][1:-1] = g[2][1:-1]
    g[-1][1:-1] = g[-3][1:-1]
    
    g[0][1:-1] = g[2][1:-1]
    g[-1][1:-1] = g[-3][1:-1]
    
    g[1:-1,0] = g[1:-1,2]
    g[1:-1,-1] = g[1:-1,-3]
    
    return g
    

def binaryfit(Img, H_phi):
    """
   [C1,C2]= binaryfit(phi,U,epsilon) computes c1 c2 for optimal binary fitting 
   input: 
       Img: input image
       phi: level set function
       epsilon: parameter for computing smooth Heaviside and dirac function
   output: 
       C1: a constant to fit the image U in the region phi>0
       C2: a constant to fit the image U in the region phi<0

    """
    a = H_phi*Img
    numer_1=np.sum(a)
    denom_1=np.sum(H_phi)
    C1 = numer_1/denom_1

    b=(1-H_phi)*Img
    numer_2=np.sum(b)
    c=1-H_phi
    denom_2=np.sum(c)
    C2 = numer_2/denom_2
    
    return C1, C2
