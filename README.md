# EcoWatch: An Adaptive Framework for Ecological Monitoring in WMSNs

EcoWatch is an open-source ecological monitoring framework designed for deployment in Wireless Multimedia Sensor Networks (WMSNs) in remote environments. Its primary purpose is to monitor bird populations through real-time detection and counting, using an energy-efficient approach to extend operational lifespans in resource-constrained settings.

![EcoWatch Framework](path_to_image.png)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

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

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SamiGandhi/Ecowatch
   cd EcoWatch
