# PCB-Defect-Detector

## Project Overview
This project implements a Printed Circuit Board (PCB) defect detection system using the YOLOv8 (You Only Look Once, version 8) object detection model. The system is designed to identify various defects on PCBs, such as missing holes, mouse bites, open circuits, shorts, spurs, and spurious copper. The model is trained on a custom dataset and can perform inference on new images, generating visual annotations and exporting detection results in PASCAL VOC XML format for further analysis or integration into other systems.

## Features
-   **YOLOv8 Training**: Train a YOLOv8 model on a custom PCB defect dataset.
-   **GPU Acceleration**: Leverages CUDA for faster training and inference (if GPU is available).
-   **Inference**: Perform object detection on new images.
-   **Visual Output**: Saves images with detected defects and bounding box annotations.
-   **PASCAL VOC XML Export**: Generates industry-standard PASCAL VOC XML annotation files for each detected image, including bounding box coordinates, class labels, and confidence scores.

## Setup and Installation
This project is designed to run in a Google Colab environment, leveraging its GPU capabilities. 

1.  **Open in Google Colab**: Upload and open the `.ipynb` notebook in Google Colab.
2.  **Mount Google Drive**: Ensure your Google Drive is mounted to access datasets and save training/inference results. The notebook assumes a dataset path like `/content/drive/MyDrive/train` and output paths within `/content/drive/MyDrive/`.
3.  **Install Dependencies**: Run the first code cell to install necessary libraries:
    ```python
    !pip install -U ultralytics opencv-python lxml
    ```

## Dataset
The model was trained on a custom PCB defect dataset. The dataset structure is expected to be:
```
/content/drive/MyDrive/train/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

The `dataset.yaml` file defines the class names and dataset paths:
```yaml
path: /content/drive/MyDrive/train
train: images
val: images
test: null

nc: 6
names:
  0: Missing_hole
  1: Mouse_bite
  2: Open_circuit
  3: Short
  4: Spur
  5: Spurious_copper
```

## Model Training
The YOLOv8 `yolov8n.pt` model was fine-tuned for 20 epochs with an image size of 640x640 and a batch size of 32 (on GPU). The training results, including the best model weights, are saved to `/content/drive/MyDrive/pcb_training_runs/yolo/weights/best.pt`.

### Training Parameters
-   **Model**: `yolov8n.pt`
-   **Epochs**: 20
-   **Image Size (`imgsz`)**: 640
-   **Batch Size**: 32 (on GPU), 16 (on CPU)
-   **Patience**: 10 (early stopping)
-   **Save Period**: 5 epochs
-   **Project Directory**: `/content/drive/MyDrive/pcb_training_runs`

## Inference
After training, the best model weights (`best.pt`) are loaded for inference. Images for inference are expected in a specified Google Drive folder (e.g., `/content/drive/MyDrive/uploaded_Images`).

### Running Inference
1.  **Load the Trained Model**:
    ```python
    from ultralytics import YOLO
    model_best = YOLO("/content/drive/MyDrive/pcb_training_runs/yolo/weights/best.pt")
    ```
2.  **Define Input and Output Directories**:
    ```python
    INPUT_IMAGE_DIR = '/content/drive/MyDrive/uploaded_Images'
    OUTPUT_RESULTS_DIR = '/content/drive/MyDrive/uploaded_Images/inference_results'
    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
    ```
3.  **Run Prediction**:
    ```python
    results = model_best.predict(
        source=INPUT_IMAGE_DIR,
        save=True,
        project=OUTPUT_RESULTS_DIR,
        name='predictions',
        device='0' if torch.cuda.is_available() else 'cpu',
        imgsz=640,
        conf=0.25
    )
    ```
    The annotated images and YOLO-format labels (`.txt`) will be saved in a subdirectory within `OUTPUT_RESULTS_DIR` (e.g., `/content/drive/MyDrive/uploaded_Images/inference_results/predictions`).

## PASCAL VOC XML Annotation Export
To provide compatibility with other vision tools, the detected bounding boxes and class information are converted into PASCAL VOC XML format. A utility function `convert_yolo_to_voc_xml` handles this conversion.

### XML Output Location
The generated XML files are saved in a dedicated `annotations_voc` subdirectory within the inference results folder (e.g., `/content/drive/MyDrive/uploaded_Images/inference_results/predictions/annotations_voc`).

## Example Workflow
1.  **Mount Google Drive**.
2.  **Install dependencies** (`ultralytics`, `opencv-python`, `lxml`).
3.  **Load the `best.pt` model**.
4.  **Place new images** you want to annotate into a Google Drive folder (e.g., `/content/drive/MyDrive/uploaded_Images`).
5.  **Run the inference code** provided in the notebook.
6.  **Execute the XML conversion code** to generate PASCAL VOC annotations.
7.  **Access results** in your specified output directory on Google Drive.

## Results
The model achieved strong performance on the validation set after 20 epochs, with a mean Average Precision (mAP50-95) of **0.589** and mAP50 of **0.989**. Individual class performance was also high:

-   **Missing_hole**: mAP50-95 = 0.599, mAP50 = 0.99
-   **Mouse_bite**: mAP50-95 = 0.568, mAP50 = 0.992
-   **Open_circuit**: mAP50-95 = 0.631, mAP50 = 0.993
-   **Short**: mAP50-95 = 0.599, mAP50 = 0.981
-   **Spur**: mAP50-95 = 0.561, mAP50 = 0.991
-   **Spurious_copper**: mAP50-95 = 0.578, mAP50 = 0.987

Inference on 9 example images from `uploaded_Images` successfully detected various defects and generated corresponding PASCAL VOC XML files.

## Contributing
Feel free to fork the repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
