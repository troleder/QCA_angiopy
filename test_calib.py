import numpy as np
import scipy.ndimage
import scipy.signal

# Simulate a frame with a diagonal catheter 20px wide
# Dark tube on white background
shape = (512, 512)
frame = np.full(shape, 200, dtype=float)
# Draw a dark tube from (100, 100) to (300, 400)
# This is angle ~56 deg from horizontal
for y in range(512):
    for x in range(512):
        # distance to line
        dist = np.abs(1.5*(x-100) - (y-100)) / np.sqrt(1.5**2 + 1)
        if dist <= 10:
            frame[y, x] = 50

cx, cy = 200, 250
L = 60
base_t = np.arange(-L, L)
bestDiam = None
bestTheta = None
bestPeaks = None

for theta_deg in range(0, 180, 5):
    theta = np.deg2rad(theta_deg)
    x_vals = cx + base_t * np.cos(theta)
    y_vals = cy + base_t * np.sin(theta)
    
    coords = np.vstack((y_vals, x_vals))
    profile = scipy.ndimage.map_coordinates(frame, coords, mode='nearest')
    
    smoothed = np.convolve(profile, np.ones(3)/3.0, mode='same')
    grad = np.abs(np.gradient(smoothed))
    if grad.max() == 0:
        continue
    
    peaks = scipy.signal.find_peaks(grad, height=grad.max() * 0.25, distance=4)[0]
    if len(peaks) >= 2:
        peak_heights = grad[peaks]
        top2_idx = np.argsort(peak_heights)[-2:]
        best_p0 = min(peaks[top2_idx])
        best_p1 = max(peaks[top2_idx])
        
        diam = best_p1 - best_p0
        if diam > 3:
            if bestDiam is None or diam < bestDiam:
                bestDiam = diam
                bestPeaks = (best_p0, best_p1)
                bestTheta = theta
                print(f"Angle {theta_deg}: diam={diam}, peaks={bestPeaks}")

print(f"Final: angle={np.rad2deg(bestTheta)}, diam={bestDiam}")
