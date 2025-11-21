# Object Detection Project with YOLO

A simple and organized object detection project using YOLOv8 for detecting objects in user-provided images.

## 📁 Project Structure

```
object_detection/
├── 01_data_preparation.ipynb    # Model download and test image preparation
├── 02_object_detection.ipynb    # Main detection implementation
├── 03_model_evaluation.ipynb    # Performance testing and evaluation
├── models/                      # Saved models
├── test_images/                 # Sample test images
├── results/                     # Detection results
└── README.md                    # This file
```

## 🚀 Quick Start

### Using Jupyter Notebooks

1. **Prepare Data and Model**
   ```bash
   jupyter notebook 01_data_preparation.ipynb
   ```
   Downloads YOLOv8 model and sample test images. Creates necessary directories.

2. **Run Object Detection**
   ```bash
   jupyter notebook 02_object_detection.ipynb
   ```
   Main detection functionality with examples and interactive functions.

3. **Evaluate Model Performance**
   ```bash
   jupyter notebook 03_model_evaluation.ipynb
   ```
   Speed testing, confidence analysis, and robustness evaluation.

## 📖 Usage Examples

### In Jupyter Notebook:
```python
# After running the notebooks, use the detector
result = detect_user_image('path/to/your/image.jpg', confidence=0.5)

# Or use the ObjectDetector class directly
detector = ObjectDetector()
results, annotated_image = detector.detect_objects('your_image.jpg')
```

## 🛠️ Requirements

- Python 3.8+
- OpenCV (`cv2`)
- PyTorch
- Ultralytics YOLO (`ultralytics`)
- Matplotlib
- NumPy
- Pillow

Install with:
```bash
pip install ultralytics opencv-python matplotlib numpy pillow
```

## 📊 Model Information

- **Model**: YOLOv8n (Nano version - fast and efficient)
- **Classes**: 80 COCO classes (person, car, dog, bicycle, etc.)
- **Input**: Images (JPG, PNG, JPEG)
- **Output**: Bounding boxes with class labels and confidence scores
- **Model Size**: ~6MB (automatically downloaded)

## 🎯 Features

- ✅ Easy setup with organized notebooks
- ✅ Pre-trained YOLO model (no training required)
- ✅ Interactive detection functions
- ✅ Confidence threshold adjustment
- ✅ Performance evaluation tools
- ✅ Speed benchmarking
- ✅ Robustness testing
- ✅ Automatic result saving with annotations
- ✅ Visual results with side-by-side comparison

## 📈 Performance Features

The evaluation notebook provides:
- **Speed Analysis**: FPS measurement and timing
- **Confidence Testing**: How different thresholds affect detection
- **Class Distribution**: What objects are most commonly detected
- **Robustness Testing**: Performance under different image conditions
- **Comprehensive Report**: Overall model performance summary

## 🤝 Usage Tips

1. **Getting Started**: Run notebooks in order (01 → 02 → 03)
2. **Image Quality**: Better lighting and resolution = better detection
3. **Confidence Tuning**: 
   - Lower (0.3) = more detections, some false positives
   - Higher (0.7) = fewer but more accurate detections
   - Default (0.5) = good balance
4. **Supported Formats**: JPG, PNG, JPEG
5. **Results**: Check `results/` folder for annotated images

## 🆘 Troubleshooting

**Model not loading?**
- Ensure internet connection for initial model download
- Check if `models/` directory exists

**No detections?**
- Try lower confidence threshold (0.3)
- Ensure image contains clear, recognizable objects
- Check if image path is correct

**Slow performance?**
- Model runs faster on GPU if available
- Use smaller images for faster processing
- First run is slower due to model initialization

**Import errors?**
- Install required packages: `pip install ultralytics opencv-python matplotlib`
- Ensure Python 3.8+ is being used

## 🔍 What Can It Detect?

The model can detect 80 different object types including:
- **People & Animals**: person, dog, cat, horse, sheep, cow, etc.
- **Vehicles**: car, motorcycle, airplane, bus, train, truck, boat
- **Everyday Objects**: chair, table, bottle, cup, book, laptop, phone
- **Food**: banana, apple, sandwich, pizza, donut, cake
- **Sports**: sports ball, tennis racket, baseball bat, skateboard
- And many more!

## 📝 Next Steps

- Experiment with different confidence thresholds
- Try your own images
- Analyze the performance metrics
- Consider upgrading to larger YOLO models for better accuracy
- Implement video detection
- Add custom object classes

## 📄 License

This project uses the YOLOv8 model from Ultralytics. Please check their license for commercial usage.
- **Everyday Objects**: chair, table, bottle, cup, book, laptop, phone
- **Food**: banana, apple, sandwich, pizza, donut, cake
- **Sports**: sports ball, tennis racket, baseball bat, skateboard
- And many more!

## 📝 Next Steps

- Experiment with different confidence thresholds
- Try your own images
- Analyze the performance metrics
- Consider upgrading to larger YOLO models for better accuracy
- Implement video detection
- Add custom object classes

## 📄 License

This project uses the YOLOv8 model from Ultralytics. Please check their license for commercial usage.
