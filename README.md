# Summit Navigator

**Summit Navigator** is a Python-based project for accurately identifying and analyzing peaks in multi-modal gray-scale histograms. It uses advanced algorithms for peak searching, merging, and thresholding to enhance image segmentation, background removal, and object detection. The project is inspired by mountain exploration tactics and employs statistical metrics to refine peak detection.

---

## **Features**
- **Peak Detection**: Accurately identifies true peaks in noisy, multi-modal histograms without prior knowledge of the number of modes or distances between them.
- **Peak Merging**: Eliminates false peaks using statistical metrics like \(R^2\) values from linear regression.
- **Thresholding**: Dynamically computes thresholds between peaks and modifies image intensities for segmentation.
- **Performance Evaluation**: Computes precision, recall, and F-measure using ground truth data for robust performance evaluation.

---

## **Applications**
- **Surface Inspection**: Detect defects in industrial components like capacitors and metal surfaces.
- **Medical Imaging**: Segment tissues, cells, and abnormalities; remove noise and irrelevant backgrounds.
- **Drone-Based Vision**: Enhance images for inspecting bridges, pipelines, and large surfaces.
- **Automation and Robotics**: Assist robots in scene understanding and object detection in vision-based systems.

---

## **Requirements**

The project uses Python and the following libraries:

```bash
opencv-python 
numpy 
matplotlib 
scikit-learn
```

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure
```bash
.
├── Images/                # Folder containing input images
├── main.py                # Main script to run the Summit Navigator
├── ground_truth.json      # JSON file containing ground truth peak data
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

## Usage
1. **Prepare Ground Truth Data**

    - Create a JSON file (`ground_truth.json`) with the ground truth peaks for your images in the following format:

    ```bash
    {
        "image.jpeg": [10, 45, 75],
        "image2.jpeg": [20, 50, 80]
    }
    ```

2. **Run the Program**

    To process an image:

    ```bash
    python main.py
    ```

**Input**:

- Place your input image in the Images/ directory and update the filename in the script (main.py).

## Output:

- Displays the original image and the transformed image with modified intensities.
- Prints the detected dominant peaks and their corresponding thresholds.

## Example Output

**Detected Peaks:**
```bash
Ground Truth Peaks: [10, 45, 75]
Dominant Peaks: [12, 46, 76]
```

**Visual Output:**

- **Original Image**: Grayscale input image.
- **Transformed Image**: Segmented image with intensity values adjusted based on detected peaks.

## Evaluation Metrics

The program evaluates its performance against ground truth data using:

- **Precision:** Fraction of correctly detected peaks among all detected peaks.
- **Recall:** Fraction of correctly detected peaks among all ground truth peaks.
- **F-Measure:** Harmonic mean of precision and recall.

## Key Algorithms

1. **Peak Searching**:
    - Detects initial peaks using histogram analysis.
    - Calculates observability indices to refine peaks.

2. **Peak Merging**:
    - Uses R2R2 from linear regression to merge false peaks.
    - Retains only significant peaks part of unimodal trends.

3. **Thresholding**:
    - Computes thresholds as minima between peaks.
    - Adjusts pixel intensities for segmentation.

## References

This project is inspired by:

- Techniques in histogram thresholding and peak detection.
- Research on sunspot datasets, fuzzy thresholding, and statistical regression models.
- [Research Paper](./Summit_Navigator_A_Novel_Approach_for_Local_Maxima_Extraction.pdf)