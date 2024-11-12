# EcoWatch: An Adaptive Framework for Ecological Monitoring in WMSNs

EcoWatch is an open-source ecological monitoring framework designed for deployment in Wireless Multimedia Sensor Networks (WMSNs) in remote environments. Its primary purpose is to monitor bird populations through real-time detection and counting, using an energy-efficient approach to extend operational lifespans in resource-constrained settings.

## Overview

EcoWatch leverages a dual-model architecture combining YOLOv8 for object detection and LTCE for accurate object counting, enabling efficient and accurate wildlife monitoring. It applies Region-of-Interest (ROI)-based video compression and uses LoRaWAN for low-power data transmission, making it suitable for long-term deployment in remote or inaccessible areas.

## Features

- **Real-Time Detection and Counting**: Uses YOLOv8 for bird detection and LTCE for accurate counting.
- **ROI-Based Video Compression**: Optimizes data usage by compressing non-essential regions of video frames.
- **LoRaWAN Transmission**: Enables long-distance, low-power data transmission.
- **Energy Efficiency**: Reduces energy consumption by up to 58.7%.
- **Bandwidth Optimization**: Reduces data transmission load by up to 69.8%.
- **Scalable and Adaptable**: Suitable for deployment in diverse ecological monitoring scenarios.

## Installation

### Prerequisites

- Python 3.8 or later
- PyTorch
- OpenCV
- Additional libraries (specified in `requirements.txt`)


1. **Clone the repository**:
   ```bash
   git clone https://github.com/SamiGandhi/Ecowatch
   cd EcoWatch
   
2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
  
## Usage

1. **Navigate to the source directory**:
   ```bash
   cd source

2. **Run the example script**:
   ```bash
   python example.py

## Running the ROI Example:
To run the ROI (Region of Interest) example, follow these steps:
1. Ensure you have cloned the repository and installed the dependencies as described above.
2. Navigate to the source directory:
   ```bash
   cd source

3. Run the ROI example script:
      ```bash
      python example.py
4. Find the trace files:   
      The trace files will be generated within a directory named after the video.
      For example, if the video is named example_video.mp4, the trace files will be located in a directory named example_video.
      Directory Structure
      ```bash
      EcoWatch/
      │
      ├── source/
      │   ├── example.py
      │   └── ...
      │
      ├── requirements.txt
      │
      └── README.md

## Parameters Description
### The parameters class contains various configuration settings for the EcoWatch project. Here is a description of each parameter:
      
      captured_video_path: Path to the captured video file.
      video_base_path: Base path of the video file (without extension).
      captured_frame_dir: Directory to store captured frames.
      masks_dir: Directory to store ROI masks.
      reference_frames_dir: Directory to store reference frames.
      roi_frame_dir: Directory to store ROI frames.
      output_directory: Directory to store output files.
      decoded_frames_dir: Directory to store decoded frames.
      fps: Frames per second (default is 2).
      default_width: Default width of the video frames (default is 144).
      default_height: Default height of the video frames (default is 144).
      gop_coefficient: Group of Pictures coefficient (default is 30).
      quality_factor: Quality factor for encoding (default is 80).
      high_quality_factor: High quality factor for encoding (default is 40).
      low_quality_factor: Low quality factor for encoding (default is 20).
      zero: Placeholder for zero value (default is 1).
      zone_size: Size of the zone (default is 8).
      DCT: Discrete Cosine Transform method (default is "CLA").
      level_numbers: Number of levels (default is 1).
      entropy_coding: Entropy coding method (default is "RLE_EG").
      trace_file_path: Path to the trace file.
      packed_load_size: Size of the packed load (default is 512).
      interleaving: Interleaving method.
      threshold: Threshold value (default is 1).
      max_level_S: Maximum level S (default is 0).
      enhance_method: Enhancement method.
      enhance_param: Enhancement parameter.
      w1: Weight parameter 1 (default is 8).
      w2: Weight parameter 2 (default is 4).
      w3: Weight parameter 3 (default is 2).
      roi_threshold: ROI threshold value (default is 10).
      running: Running mode.
      POWER: Power consumption (default is 0.023).
      PROC_CLOCK: Processor clock speed (default is 7.3728).
      CAPTURE_E_PER_BLOCK: Energy per block for capture (default is 2.65e-6).
      CYCLES_PER_DIV: Cycles per division (default is 300).
      CYCLES_PER_ADD: Cycles per addition (default is 1).
      CYCLES_PER_FADD: Cycles per floating-point addition (default is 20).
      CYCLES_PER_SHIFT: Cycles per shift (default is 1).
      CYCLES_PER_MUL: Cycles per multiplication (default is 2).
      CYCLES_PER_FMUL: Cycles per floating-point multiplication (default is 100).
      CLA_ADD_NB: Number of additions in CLA (default is 464).
      CLA_MUL_NB: Number of multiplications in CLA (default is 192).
      MAX_S_LAYERS: Maximum number of S layers (default is 5).
      DISTANCE: Distance in meters (default is 100).
      FREQUENCY: Frequency in Hz (default is 868e6).
      ENVIRONEMENT: Environment type (default is "wildlife_zone").
      HUMIDITY_LEVEL: Humidity level in percent (default is 70).
      VEGETATION_DENSITY_LEVEL: Vegetation density level (default is 5).
      scale_factor: Scale factor (default is 100).

      
## Results
In order to obtain the results, please contact one of the authors within the contact section.


## Contributing
We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your branch to your fork.
5. Create a pull request.


## Contact
