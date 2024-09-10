## Frequency-Domain Popularity Forecasting with Shape-Based Retrieval (FPF-SR)

This repository contains the implementation of **FPF-SR**, a model designed for **popularity forecasting** using **shape-based sequence retrieval** and **frequency domain analysis**. The model helps predict the popularity of social media content by leveraging historical data and global trend features.

### Key Features

- **Shape-Based Sequence Retrieval**: Efficiently retrieves the most relevant historical data based on the shape of short input sequences.
- **Frequency Domain Analysis**: Captures global temporal patterns by analyzing the retrieved data in the frequency domain using techniques like Fourier transforms.
- **Multi-Point Trend Forecasting**: Enhances single-point prediction by integrating multi-point forecasting, which leads to improved accuracy, especially for short and sparse data.

### Prerequisites

- Python 3.8 or later
- Required libraries:
  - NumPy
  - SciPy
  - PyTorch
  - Matplotlib

To install all the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
