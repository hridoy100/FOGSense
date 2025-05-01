# This is the main repository of the code behind the paper: https://arxiv.org/pdf/2411.11764

# Detailed Webpage view: https://hridoy100.github.io/FOGSense/

# Parkinson's FOG Detection

Neural network system for Parkinson's Freezing of Gait detection with federated learning and resource monitoring.

## Project Structure

```    
parkinsons-fog-detection/
├── data/
│   ├── csv/                      # Raw and processed CSV files
│   ├── mean_subtract/            # Processed data directory
│   │   └── gaf_images/          # GAF images for CNN input
│   │       ├── AccV/            # Vertical acceleration
│   │       ├── AccML/           # Medio-lateral acceleration
│   │       └── AccAP/           # Anterior-posterior acceleration
│   └── federated_learning_data/  # Data split for federated learning
├── models/
│   └── cnn_models.py            # CNN architecture definitions
└── notebooks/
    ├── preprocess-for-cnn.ipynb              # Data preprocessing
    ├── image_training_with_multi_channel_cnn.ipynb              # Non-Federated image training
    ├── federated_learning_weighted_avg.ipynb
    └── federated_learning_with_resource_monitor.ipynb
```

## Features

### Core Functionality
- Multi-channel CNN for FOG detection
- Federated learning with weighted averaging
- GAF transformation of accelerometer data

### Resource Monitoring
- Real-time tracking of:
  - CPU utilization
  - Memory usage
  - GPU utilization and memory
  - Training metrics
- Visual monitoring through:
  - Live resource usage plots
  - Training metrics visualization
  - Per-class performance tracking

## Requirements

```
tensorflow>=2.0
numpy
pandas
pyts
psutil
gputil
matplotlib
seaborn
```

## Usage

### Resource Monitoring Setup
```python
monitor = ResourceMonitor(interval=1.0)
monitor_callback = ResourceMonitorCallback(monitor)

# Add to training callbacks
callbacks = [
    monitor_callback,
    MetricsDisplayCallback(num_classes=2)
]
```

### Federated Training with Monitoring
```python
server = create_federated_learning_system(
    num_clients=5,
    train_generator=train_generator,
    valid_generator=valid_generator
)

monitor.start()
metrics_history = train_federated(
    server,
    num_rounds=10,
    local_epochs=5,
    callbacks=callbacks
)
```

### Resource Analysis
```python
# View resource usage summary
monitor.plot_resources()
summary = monitor.get_summary()
```

## Model Architecture

- Three CNN branches (AccV, AccML, AccAP)
- Each branch:
  - 3 Conv2D layers (32->64->128 filters)
  - Batch normalization
  - MaxPooling and Dropout
- Dense layers: 128->64->2
- Softmax output

## Dataset
- 62 subjects split:
  - Training: 69.4%
  - Testing: 19.4%
  - Validation: 11.3%

## Performance Analysis
- Accuracy: 86.99%
- Precision: 86.84%
- Recall: 87.00%
- Resource utilization tracked per training round

### Model Performance Comparison
```
┌─────────────────┬──────────┬───────────┬────────┬──────────┬─────────────┐
│ Model Type      │ Accuracy │ Precision │ Recall │ F1 Score │ Specificity │
├─────────────────┼──────────┼───────────┼────────┼──────────┼─────────────┤
│ Federated CNN   │ 86.99%   │ 86.84%    │ 87.00% │ 86.86%   │ 76.71%     │
└─────────────────┴──────────┴───────────┴────────┴──────────┴─────────────┘
```

### Detailed Metrics for Federated Learning

```
Class-wise Performance:
┌────────┬───────────┬─────────┬──────────┐
│ Class  │ Precision │ Recall  │ F1 Score │
├────────┼───────────┼─────────┼──────────┤
│ Normal │ 82.41%    │ 76.71%  │ 79.45%   │
│ FOG    │ 89.00%    │ 92.01%  │ 90.48%   │
└────────┴───────────┴─────────┴──────────┘
```

### Resource Utilization Pattern
```
Training Resource Usage:    
                                              
Memory   ┤█████████░░ 75%                     
GPU Mem  ┤████████░░░ 70%                     
CPU      ┤███████░░░░ 65%                     
         └─────────────────────────
              25   50   75   100%
```

### Additional Performance Metrics
- True Positives: 61.82%
- True Negatives: 25.16%
- False Positives: 7.64%
- False Negatives: 5.37%

## Acknowledgments
- TDCSFOG dataset contributors
- Parkinson's research community
