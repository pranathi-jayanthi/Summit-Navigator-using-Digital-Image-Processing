import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def getImage(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def getHistogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    return hist, bins


def plotHistogram(image, bins):
    fig = plt.figure(figsize=(10, 7))

    plt.hist(image, bins=bins, range=[0, 256])
    plt.title("Histogram")
    plt.show()


def getSmoothHistogram(hist, window_size=3):
    kernel = np.ones(window_size) / window_size
    smoothed_hist = np.convolve(hist, kernel, mode='same')
    return smoothed_hist


def getInitialPeak(hist):
    S = []
    for pos in range(1, len(hist)):
        if hist[pos] > hist[pos - 1] and hist[pos] > hist[pos + 1]:
            S.append(pos)

    return S


def peakSearching(hist, initial_peaks, epsilon=1e-6):

    P = len(initial_peaks)
    L_tmp = []
    observability_indices = np.zeros(P)

    for k in range(1, P):
        sk = initial_peaks[k]
        hk = hist[sk]

        xk = sk*np.ones(k)
        yk = hk*np.ones(k)

        deltaXk = xk - initial_peaks[:k]
        deltaYk = yk - hist[initial_peaks[:k]] + epsilon

        lk = xk - (yk * deltaXk)/deltaYk
        L_tmp.append(min(lk))

    L_s = np.min(L_tmp)

    for k in range(1, P):
        sk = initial_peaks[k]

        observability_indices[k] = hist[sk] / (sk-L_s)

    dominant_peaks = [initial_peaks[0]]

    for k in range(1, P - 1):
        if observability_indices[k] > observability_indices[k - 1] and observability_indices[k] > observability_indices[k + 1]:
            dominant_peaks.append(initial_peaks[k])

    if observability_indices[-1] > observability_indices[-2]:
        dominant_peaks.append(initial_peaks[-1])

    return dominant_peaks


def calculate_r_squared_least_squares(histogram, peak1, peak2):
    x = np.arange(peak1, peak2 + 1)
    y = histogram[peak1:peak2 + 1]

    model = np.polyfit(x, y, 1)
    y_fit = np.polyval(model, x)

    return r2_score(y, y_fit)


def peakMerging(dominant_peaks, hist, r_squared_threshold=0.8):
    cnt = 1
    while (cnt > 0):
        d = []
        r_squared = []
        for k in range(len(dominant_peaks) - 1):
            r_squared.append(calculate_r_squared_least_squares(
                hist, dominant_peaks[k], dominant_peaks[k+1]))

        for i in range(1, len(dominant_peaks) - 1):
            if r_squared[i] < r_squared_threshold and ((i > 1 and hist[dominant_peaks[i]] < hist[dominant_peaks[i - 1]]) or hist[dominant_peaks[i]] < hist[dominant_peaks[i + 1]]):
                d.append(dominant_peaks[i])

        cnt = len(d)
        if cnt > 0:
            for i in d:
                dominant_peaks.remove(i)

    return dominant_peaks


def threshold(dominant_peaks, histogram):
    thr = []
    for i in range(len(dominant_peaks) - 1):
        temp = histogram[dominant_peaks[i]: dominant_peaks[i + 1] + 1]
        mini = min(temp)
        indexes = [dominant_peaks[i] + index for index,
                   element in enumerate(temp) if element == mini]
        thr.append(np.median(indexes))
    return thr


def distribute_image(image, thresholds):
    thresholds = [0] + thresholds + [255]
    output_image = np.zeros_like(image)

    for i in range(1, len(thresholds)):
        if i == 0:
            mask = np.where(image < thresholds[0])
            output_image[mask] = 0
        elif i == len(thresholds) - 1:
            mask = np.where(image >= thresholds[i])
            output_image[mask] = 255
        else:
            lower_bound = thresholds[i - 1]
            upper_bound = thresholds[i]
            mask = np.where((image >= lower_bound) & (image < upper_bound))
            output_image[mask] = np.round((thresholds[i] + thresholds[i-1])/2)

    return output_image


def calculate_metrics(ground_truth, detected_peaks, total_classes=1):
    tp = len(set(ground_truth).intersection(detected_peaks)) 
    fp = len(set(detected_peaks) - set(ground_truth))        
    fn = len(set(ground_truth) - set(detected_peaks))        
    print(tp,fp,fn)

    if total_classes > 1:
        precision = sum(tp / (tp + fp) if (tp + fp) !=
                        0 else 0 for _ in range(total_classes)) / total_classes
        recall = sum(tp / (tp + fn) if (tp + fn) !=
                     0 else 0 for _ in range(total_classes)) / total_classes
        f_measure = (2 * precision * recall) / (precision +
                                                recall) if (precision + recall) != 0 else 0
    else:
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f_measure = (2 * precision * recall) / (precision +
                                                recall) if (precision + recall) != 0 else 0

    return {"Precision": precision, "Recall": recall, "F-Measure": f_measure}


if __name__ == "__main__":
    filename = 'Images/image.jpeg'
    ground_truth_file = 'ground_truth.json'  

    
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    ground_truth_peaks = ground_truth_data.get(filename.split("/")[1], []) 
    print("Ground Truth Peaks : ",ground_truth_peaks)

    image = getImage(filename)
    hist, bins = getHistogram(image)
    hist = getSmoothHistogram(hist)
    initialPeak = getInitialPeak(hist)
    dominant_peaks = peakSearching(hist, initialPeak)
    dominant_peaks = peakMerging(dominant_peaks, hist)
    thresholds = threshold(dominant_peaks, hist)
    thresholds = [int(i) for i in thresholds]

    print("Dominants Peaks : ",thresholds)

    output_image = distribute_image(image, thresholds)
    distributed_image = cv2.applyColorMap(output_image, cv2.COLORMAP_PARULA)
    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    distributed_image_rgb = cv2.cvtColor(distributed_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(distributed_image_rgb)
    plt.title("Transformed Image")
    plt.axis("off")
    
    plt.show()

