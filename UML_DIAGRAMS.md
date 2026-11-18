# Comprehensive UML Diagrams Specification
## Project: Automated Road Distress Detection and Severity Mapping Using YOLOv8 and Computer Vision
**Academic Year:** Final-Year (7th/8th Semester) B.Tech / B.E. Computer Science & Engineering Capstone Project  
**Tech Stack:** Python 3.10+, Ultralytics YOLOv8, OpenCV 4.x, Streamlit, Pandas, Green's Theorem Morphology

---

## 1. Use Case Diagram
The Use Case Diagram depicts the functional interactions between primary system actors (Highway Maintenance Surveyor, Systems Analyst) and system automated subsystems.

```mermaid
flowchart LR
    subgraph Actors ["External Actors"]
        Surveyor["Highway Surveyor / Field Engineer"]
        Analyst["Road Asset Manager / Analyst"]
        Camera["Inspection Vehicle Camera / Media Feed"]
    end

    subgraph System ["Road Distress Detection System Boundary"]
        UC1["UC-01: Ingest Road Video/Image Stream"]
        UC2["UC-02: Select Model Weights & Configure Confidence"]
        UC3["UC-03: Tune Ground-Plane Spatial Calibration (k)"]
        UC4["UC-04: Execute YOLOv8 Distress Localization"]
        UC5["UC-05: Run CV Morphology & Green's Theorem Area"]
        UC6["UC-06: Map Damage Severity Tier (Low/Med/High)"]
        UC7["UC-07: Display Real-Time HUD Overlay & Live FPS"]
        UC8["UC-08: View Microscopic ROI Inspection Gallery"]
        UC9["UC-09: Analyze Statistical Trends & PCI Score"]
        UC10["UC-10: Plot Geospatial Damage GPS Route Map"]
        UC11["UC-11: Export Telemetry Records to CSV Report"]
    end

    Camera --> UC1
    Surveyor --> UC1
    Surveyor --> UC2
    Surveyor --> UC3
    Surveyor --> UC7
    Surveyor --> UC8

    UC1 --> UC4
    UC4 --> UC5
    UC5 --> UC6
    UC6 --> UC7

    Analyst --> UC9
    Analyst --> UC10
    Analyst --> UC11
```

---

## 2. Class Diagram
The Class Diagram illustrates the object-oriented structure, class attributes, method signatures, data types, and structural relationships.

```mermaid
classDiagram
    class RoadDistressDetector {
        +Dict~int, str~ DISTRESS_CLASSES
        +str model_path
        +float conf_thresh
        +Any model
        +bool is_custom_distress_model
        +__init__(model_path: str, conf_thresh: float)
        -_load_model() void
        +detect(frame: ndarray, conf_threshold: float) List~Dict~
        -_extract_road_anomalies(frame: ndarray, base_conf: float) List~Dict~
        -_apply_nms(detections: List~Dict~, iou_thresh: float) List~Dict~
    }

    class SeverityMeasurer {
        +float LOW_SEVERITY_LIMIT
        +float HIGH_SEVERITY_LIMIT
        +Dict COLOR_MAP
        +float k
        +__init__(spatial_calibration_k: float)
        +update_calibration_k(new_k: float) void
        +process_roi(frame: ndarray, box: Tuple, blur_kernel: Tuple, canny_thresh1: int, canny_thresh2: int) Dict~str, Any~
        +classify_severity(metric_area_cm2: float) str
        +annotate_frame(frame: ndarray, detection: Dict, measurement: Dict, draw_hud: bool) ndarray
        -_empty_result(box: Tuple) Dict~str, Any~
    }

    class StreamlitDashboardApp {
        +List~Dict~ telemetry_records
        +int total_frames_processed
        +render_sidebar_controls() Tuple
        +render_live_processing_hub() void
        +render_analytical_insights() void
        +render_system_documentation() void
        +reset_telemetry() void
    }

    class DetectionRecord {
        +str timestamp
        +int frame_idx
        +str distress_class
        +float confidence
        +Tuple~int, int, int, int~ bounding_box
        +float pixel_area
        +float metric_area_cm2
        +str severity_tier
        +float latitude
        +float longitude
    }

    class UltralyticsYOLOEngine {
        +predict(source: ndarray, conf: float, verbose: bool) List
        +save(filename: str) void
    }

    class OpenCVVisionEngine {
        +cvtColor(src: ndarray, code: int) ndarray
        +GaussianBlur(src: ndarray, ksize: Tuple, sigma: float) ndarray
        +Canny(src: ndarray, thresh1: int, thresh2: int) ndarray
        +findContours(image: ndarray, mode: int, method: int) Tuple
        +contourArea(contour: ndarray) float
    }

    StreamlitDashboardApp o-- RoadDistressDetector : uses
    StreamlitDashboardApp o-- SeverityMeasurer : uses
    StreamlitDashboardApp *-- DetectionRecord : manages session state
    RoadDistressDetector ..> UltralyticsYOLOEngine : wraps
    SeverityMeasurer ..> OpenCVVisionEngine : delegates morphology
    RoadDistressDetector ..> OpenCVVisionEngine : adaptive fallback
```

---

## 3. Sequence Diagram (Frame Ingestion & Severity Mapping)
The Sequence Diagram models the chronological message and control exchanges during live video ingestion and frame processing.

```mermaid
sequenceDiagram
    autonumber
    actor Surveyor as Highway Surveyor
    participant UI as Streamlit Dashboard (app.py)
    participant Detector as RoadDistressDetector (detector.py)
    participant YOLO as Ultralytics YOLOv8 Model
    participant Measurer as SeverityMeasurer (measurer.py)
    participant CV as OpenCV Engine
    participant State as Session State Telemetry

    Surveyor->>UI: Select Video Source & Click "Start Video Analysis"
    loop Frame Ingestion Loop (For Each Frame)
        UI->>UI: cv2.VideoCapture.read()
        UI->>Detector: detect(frame, conf_threshold)
        
        alt YOLO Weights Loaded
            Detector->>YOLO: predict(frame, conf)
            YOLO-->>Detector: Bounding Boxes, Confidence, Class IDs
        else Zero Detections / Fallback
            Detector->>CV: CLAHE + Adaptive Gaussian Filtering
            Detector->>CV: Otsu Multi-Thresholding & Canny
            CV-->>Detector: Structural Candidate Contours
        end

        Detector-->>UI: List of Detection Candidates [x1, y1, x2, y2, Class, Conf]

        loop For Each Detected Candidate
            UI->>Measurer: process_roi(frame, box)
            Measurer->>Measurer: Crop Sub-array ROI = frame[y1:y2, x1:x2]
            Measurer->>CV: cvtColor(BGR2GRAY) & GaussianBlur(5,5)
            Measurer->>CV: Canny(50, 150) & Dilation
            Measurer->>CV: findContours() & contourArea() (Green's Theorem)
            CV-->>Measurer: Summed Pixel Area (px)
            Measurer->>Measurer: Calculate Metric Area = pixel_area * (k ^ 2)
            Measurer->>Measurer: classify_severity(metric_area_cm2)
            Measurer-->>UI: ROI Analysis Dict (cm2, Severity Tier, Color, Edge Map)
            
            UI->>Measurer: annotate_frame(frame, detection, measurement)
            Measurer-->>UI: Annotated Frame with Corner Brackets & HUD Badge
            UI->>State: Append Telemetry Record (Time, Class, Area, Severity, GPS)
        end

        UI->>UI: Update Video Viewport & Real-Time KPI Cards (FPS, Damage Area)
    end
    UI-->>Surveyor: Ingestion Complete Notification & Telemetry Available
```

---

## 4. Activity Diagram (Decision Flow & Morphological Pipeline)
The Activity Diagram details the algorithmic control flow, decision branches, and mathematical computation steps.

```mermaid
flowchart TD
    Start(["Start Video/Image Ingestion"]) --> ReadFrame["Read Input Frame via OpenCV"]
    ReadFrame --> CheckValid{"Frame Valid & Non-Empty?"}
    
    CheckValid -- No --> SkipFrame["Log Warning & Advance Stream"]
    SkipFrame --> NextFrame{"More Frames in Stream?"}
    
    CheckValid -- Yes --> YOLOInfer["Run YOLOv8 Model Inference"]
    YOLOInfer --> CheckYOLO{"Detections Found?"}
    
    CheckYOLO -- Yes --> ParseBoxes["Extract Raw Bounding Boxes (x1, y1, x2, y2)"]
    CheckYOLO -- No --> VisionFallback["Execute Adaptive Vision Morphology"]
    
    VisionFallback --> CLAHE["Apply CLAHE Contrast Equalization"]
    CLAHE --> OtsuThresh["Compute Otsu + Adaptive Gaussian Threshold"]
    OtsuThresh --> CannyExtract["Extract High-Gradient Canny Edges"]
    CannyExtract --> DilateBridge["Morphological Dilation (Bridge Fissures)"]
    DilateBridge --> FilterContour["Filter Non-Road & Pebble Noise Contours"]
    FilterContour --> GeometryClassify["Classify Anomaly (Pothole / Alligator / Crack)"]
    GeometryClassify --> ParseBoxes

    ParseBoxes --> ApplyNMS["Apply Non-Maximum Suppression (IoU Threshold = 0.45)"]
    ApplyNMS --> ForEachBox["Iterate Each Isolated Distress ROI"]

    ForEachBox --> CropROI["Dynamically Crop Sub-Array ROI: frame[y1:y2, x1:x2]"]
    CropROI --> GaussianSmooth["2D Gaussian Kernel (Suppress Asphalt Granular Noise)"]
    GaussianSmooth --> EdgeHysteresis["Canny Hysteresis Dual-Thresholding"]
    EdgeHysteresis --> ExtractContour["Extract External Planar Contours"]
    ExtractContour --> GreensIntegral["Compute Contour Area via Green's Theorem: A = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)"]
    GreensIntegral --> SpatialScale["Apply Ground-Plane Scale: Metric Area = Pixel Area * k^2 (cm^2)"]
    
    SpatialScale --> SeverityCheck{"Evaluate Severity Matrix Thresholds"}
    SeverityCheck -- "Area < 150 cm²" --> LowSev["Classify: Low Severity (Green HUD #00E676)"]
    SeverityCheck -- "150 <= Area < 500 cm²" --> MedSev["Classify: Medium Severity (Orange HUD #FF9100)"]
    SeverityCheck -- "Area >= 500 cm²" --> HighSev["Classify: High Severity (Crimson Red HUD #FF1744)"]

    LowSev --> OverlayHUD["Draw Corner Brackets, Telemetry Badge & Overlay"]
    MedSev --> OverlayHUD
    HighSev --> OverlayHUD

    OverlayHUD --> LogTelemetry["Append Session Telemetry Event (Timestamp, Area, GPS, Class)"]
    LogTelemetry --> MoreBoxes{"More Boxes in Frame?"}
    MoreBoxes -- Yes --> ForEachBox
    MoreBoxes -- No --> RenderView["Render Processed Frame & Update Telemetry KPIs"]

    RenderView --> NextFrame
    NextFrame -- Yes --> ReadFrame
    NextFrame -- No --> End(["End Stream Processing & Display Analytics"])
```

---

## 5. Component Diagram
The Component Diagram shows the high-level structural decomposition into reusable, modular software units.

```mermaid
flowchart TB
    subgraph UI_Layer ["Presentation & Interaction Tier (Streamlit)"]
        StreamlitApp["app.py (Web Application Dashboard)"]
        LiveHub["Live Processing Hub Component"]
        AnalyticsHub["Analytical Insights Component"]
        DocHub["System Documentation Component"]
    end

    subgraph Logic_Layer ["Analytical & Processing Core"]
        DetectorComp["detector.py (RoadDistressDetector)"]
        MeasurerComp["measurer.py (SeverityMeasurer)"]
        SampleGenComp["generate_samples.py (Synthetic Media Generator)"]
        TestComp["test_pipeline.py (Automated Test Suite)"]
    end

    subgraph Engine_Layer ["Underlying Computer Vision & Deep Learning Engines"]
        UltralyticsYOLO["Ultralytics YOLOv8 Runtime (PyTorch)"]
        OpenCVLib["OpenCV (cv2) Image Processing Library"]
        PandasEngine["Pandas Telemetry & Data Frame Cache"]
        KaTeXEngine["LaTeX / MathJax KaTeX Renderer"]
    end

    subgraph Data_Layer ["Physical Asset & Storage Layer"]
        ModelWeights[("Model Weights: pothole_yolov8n.pt / yolov8n.pt")]
        InputMedia[("Input Feeds: Video .mp4 / Images .jpg, .png")]
        CSVReport[("Exported Telemetry CSV Reports")]
    end

    StreamlitApp --> LiveHub
    StreamlitApp --> AnalyticsHub
    StreamlitApp --> DocHub

    LiveHub --> DetectorComp
    LiveHub --> MeasurerComp
    AnalyticsHub --> PandasEngine

    DetectorComp --> UltralyticsYOLO
    DetectorComp --> OpenCVLib
    UltralyticsYOLO --> ModelWeights
    
    MeasurerComp --> OpenCVLib
    MeasurerComp --> PandasEngine

    InputMedia --> LiveHub
    PandasEngine --> CSVReport
    DocHub --> KaTeXEngine
```

---

## 6. State Machine Diagram (Road Distress Lifecycle)
The State Machine Diagram traces the state transitions of an identified road pavement anomaly from camera capture to maintenance ticketing.

```mermaid
stateDiagram-v2
    [*] --> PavementUninspected : Vehicle Traversing Roadway
    
    PavementUninspected --> IngestedFrame : Camera Sensor Ingestion
    IngestedFrame --> AnomalyDetected : YOLOv8 Feature Extraction
    
    state AnomalyDetected {
        [*] --> BoundingBoxLocalized
        BoundingBoxLocalized --> ROIExtracted : Sub-array Cropping
        ROIExtracted --> NoiseSuppressed : Gaussian Filtering (5x5)
        NoiseSuppressed --> EdgeContoursIsolated : Canny Hysteresis & Morphological Dilation
        EdgeContoursIsolated --> SurfaceAreaIntegrated : Green's Theorem Line Integration
        SurfaceAreaIntegrated --> MetricCalibrated : Apply Spatial Factor (k^2)
    }

    AnomalyDetected --> LowSeverityState : Metric Area < 150 cm²
    AnomalyDetected --> MediumSeverityState : 150 cm² <= Metric Area < 500 cm²
    AnomalyDetected --> HighSeverityState : Metric Area >= 500 cm²

    LowSeverityState --> TelemetryArchived : Tag Green HUD (#00E676) & Routine Crack Seal
    MediumSeverityState --> TelemetryArchived : Tag Orange HUD (#FF9100) & Scheduled Patching
    HighSeverityState --> CriticalAlertState : Immediate Structural Crater Hazard Detected
    
    CriticalAlertState --> TelemetryArchived : Tag Crimson HUD (#FF1744) & Emergency Dispatch
    
    TelemetryArchived --> ReportGenerated : Export CSV Telemetry Log & Geospatial Map
    ReportGenerated --> [*]
```

---

## 7. Deployment Diagram
The Deployment Diagram models the hardware topology and software deployment environment.

```mermaid
flowchart TD
    subgraph EdgeDevice ["Mobile Road Inspection Vehicle Unit"]
        CameraDevice["HD Dashboard Camera / Drone Gimbal (1080p / 4K @ 30-60 FPS)"]
        GPSModule["USB / NMEA GPS Receiver Module (Geospatial Coordinates)"]
    end

    subgraph HostWorkstation ["Edge Host Workstation / Cloud Server (Windows 11 / Linux)"]
        subgraph PythonEnv ["Python 3.10+ Project Virtual Runtime Environment"]
            StreamlitServer["Streamlit Web Server (Port :8501, Uvicorn Backend)"]
            PyTorchRuntime["PyTorch 2.x CUDA/CPU Deep Learning Runtime"]
            OpenCVRuntime["OpenCV Headless Computer Vision C++ Binaries"]
            PandasLib["Pandas In-Memory DataFrame Session Engine"]
        end

        subgraph LocalStorage ["Local File System Storage"]
            WeightFiles[("pothole_yolov8n.pt / yolov8n.pt Weights (6.2 MB)")]
            SampleMedia[("sample_road.mp4 / sample_road.jpg Media")]
            ReportDir[("telemetry_reports / CSV Data Logs")]
        end
    end

    subgraph ClientTiers ["Client Access Tier"]
        BrowserClient["Client Web Browser (Chrome / Edge / Firefox / Safari)"]
        MobileClient["Field Engineer Tablet / In-Vehicle Display Terminal"]
    end

    CameraDevice -->|"USB 3.0 / RTSP H.264 Video Stream"| OpenCVRuntime
    GPSModule -->|"Serial NMEA GPS Sentence Stream"| PandasLib

    PyTorchRuntime <-->|"Read Weights / Cache Checkpoints"| WeightFiles
    OpenCVRuntime <-->|"Read/Write Local Inspection Frames"| SampleMedia
    PandasLib -->|"Save Processed Data Logs"| ReportDir

    StreamlitServer -->|"HTTP / WebSockets (Port :8501)"| BrowserClient
    StreamlitServer -->|"HTTP / WebSockets (Port :8501)"| MobileClient
```
