
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_clusters(labels, frames_folder, num_patches=196):
    T = len(os.listdir(frames_folder))
    P = num_patches
    labels = labels.reshape(T, P)

    for i, fname in enumerate(sorted(os.listdir(frames_folder))):
        frame = cv2.imread(os.path.join(frames_folder, fname))
        frame = cv2.resize(frame, (224, 224))
        mask = labels[i].reshape(14, 14)
        mask = cv2.resize(mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
        colored_mask = cv2.applyColorMap(mask * 20, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
        cv2.imwrite(f"output_{i:04d}.jpg", overlay)
