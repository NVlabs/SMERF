import os
import cv2
import torch
import numpy as np
from math import factorial
from pyquaternion import Quaternion


def bezier_prediction_decode(lanes, n_points=11):
    lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

    def comb(n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))
    n_control = lanes.shape[1]
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
    bezier_A = torch.tensor(A, dtype=torch.float32)
    lanes = torch.tensor(lanes, dtype=torch.float32)
    lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
    lanes = lanes.numpy()

    return lanes

def points_prediction_decode(lanes, n_points=11):
    lanes = lanes.reshape(-1, n_points, 3)
    return lanes