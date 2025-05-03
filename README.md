### Breaking is Bad: Basic Crack Detection Using Image Analysis

### Project Overview
This project implements a traditional image processing pipeline to detect cracks in structural surfaces (e.g., walls, ceilings, bridges) without relying on neural networks. It focuses on preprocessing, segmentation, and analysis of cracks using techniques like thresholding, morphological operations, and heuristic classification.  

The goal is to automate structural health monitoring for civil engineering applications, providing metrics like Intersection over Union (IoU) for accuracy and normalized crack lengths for severity assessment.  


### Key Features
- **Crack Detection Workflow**:  
  - Gaussian blurring, Sobel edge detection, and thresholding for preprocessing.  
  - Morphological operations (closing, thinning) to refine crack masks.  
  - Connected component analysis and feature extraction (area, aspect ratio, circularity).  
- **Heuristic Classifier**: Filters non-crack regions using weighted thresholds on features.  
- **Metrics**:  
  - **IoU Scores** for evaluating segmentation accuracy.  
  - **Normalized Crack Lengths** for structural health insights.  
- **Dataset**:  
  - 10 real-world crack images collected from Germany and India (walls, ceilings, parking lots).  
  - Ground truth masks annotated manually.  

---

### Usage  
1. **Run Crack Detection**:  
   ```bash  
   python crack_detector.py  
   ```  
   - Modify the `image_path` variable in `crack_detector.py` to point to your input image.  
2. **Evaluate IoU**:  
   ```bash  
   python iou.py  
   ```  
   - Update paths to ground truth and detected mask images in `iou.py`.  

---

### Dataset  
- **Structure**:  
  - `input_images/`: Raw images of cracks (e.g., `crack1.png`, `crack2.png`).  
  - `ground_truth/`: Annotated masks (binary images with cracks labeled as 255).  
- **Statistics**:  
  - Total images: 10  
  - Train/test split: 80%/20%  
  - Example crack locations: Delhi, Weimar, Kerala.  

---

### Results  
- **IoU Scores**:  
  - `crack2.png`: **0.8768**  
  - `crack4.png`: **0.8594**  
- **Normalized Crack Lengths**:  
  - Crack lengths range from `9.4e-03` to `4.8e-02` (normalized units).  

---

### Strengths & Limitations  
**Strengths**:  
- Robust preprocessing (Gaussian blur, Sobel gradients) for noise reduction.  
- Feature-based filtering improves accuracy (area, aspect ratio, circularity).  
- Lightweight and interpretable heuristic rules.  

**Limitations**:  
- Sensitive to noise in low-quality images.  
- Requires manual tuning of thresholds and hyperparameters.  
- Simpler feature set compared to machine learning models.  

---

### Contributing  
Contributions are welcome! For major changes, please open an issue first to discuss what you’d like to improve.
---

### Explanation of Key Files  
1. **`crack_detector.py`**:  
   - Full pipeline for crack segmentation, feature extraction, and visualization.  
   - Uses heuristic rules to filter non-crack regions (e.g., shadows).  
2. **`iou.py`**:  
   - Computes Intersection over Union between predicted masks and ground truth.  
3. **`Report.pdf`**:  
   - Detailed documentation of methodology, results, and analysis.  

---

### Why This Matters  
This project demonstrates how traditional image processing can solve real-world problems in civil engineering, such as automated crack detection for infrastructure safety. It avoids reliance on deep learning, making it lightweight and interpretable for resource-constrained scenarios.  

--- 

### Next Steps  
- Add support for batch processing of multiple images.  
- Integrate more advanced feature engineering (e.g., curvature, branching).  
- Compare with machine learning baselines (e.g., SVM, Random Forest).  

### License  
MIT License  
