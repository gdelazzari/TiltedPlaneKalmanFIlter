import numpy as np
from typing import Callable, Optional


class KalmanFilter(object):
    def __init__(self, x0, P0, F, G, N, H):
        self.x = np.array(x0)
        self.P = np.array(P0)
        
        self.F = np.array(F)
        self.G = np.array(G)
        self.N = np.array(N)
        self.H = np.array(H)
    
    def time_update(self, Q: np.ndarray, u: np.ndarray):
        Q = np.array(Q) # covariance matrix of the noise w(t) injected into the system state through N
        u = np.array(u) # known input vector
        
        # update the state estimate
        self.x = self.F @ self.x + self.G @ u
        self.P = self.F @ self.P @ self.F.T + self.N @ Q @ self.N.T
    
    def measurement_update(self, z: np.ndarray, R: np.ndarray):
        z = np.array(z) # measured output vector
        R = np.array(R) # covariance matrix of the measurement noise
        
        # compute the Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        
        # update the state estimate
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = self.P - K @ self.H @ self.P
    
    def current_estimate(self) -> np.ndarray:
        return self.x, self.P
