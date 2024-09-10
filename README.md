# Frequency-Domain Popularity Forecasting with Shape-Based Retrieval (FPF-SR)


This repository contains the implementation of **FPF-SR**, a model designed for **popularity forecasting** using **shape-based sequence retrieval** and **frequency domain analysis**. The model helps predict the popularity of social media content by leveraging historical data and global trend features.


## Key Features


- **Shape-Based Sequence Retrieval**: Efficiently retrieves the most relevant historical data based on the shape of short input sequences.
- **Frequency Domain Analysis**: Captures global temporal patterns by analyzing the retrieved data in the frequency domain using techniques like Fourier transforms.
- **Multi-Point Trend Forecasting**: Enhances single-point prediction by integrating multi-point forecasting, which leads to improved accuracy, especially for short and sparse data.


## Prerequisites


- Python 3.10
- Required libraries:
  - NumPy
  - SciPy
  - PyTorch
  - Matplotlib


To install all the required dependencies, you can use the following command:


```bash
pip install -r requirements.txt
```

## How to Use

### 1. Download the datasets:

- [Weibo2016 Dataset](https://github.com/CaoQi92/DeepHawkes)
- [Twitter Dataset](https://github.com/CaoQi92/PREP)

### 2. Running the code: 

```bash
python run.py --dataset weibo2021 --epochs 100
```

## Project Structure

The following is an overview of the project structure:

```bash
FPF-SR/
│
├── data/                 # Datasets
├── models/               # Model architectures
├── results/              # Output and results
├── run.py              # Training script 
├── requirements.txt      # Required packages
└── README.md             # Project information
```

## Citation

If you find this work useful, please consider citing:

@inproceedings{guan2024FPF-SR,
  title={Frequency-Domain Popularity Forecasting with Shape-Based Retrieval},
  author={Canhua Guan, Zongxia Xie, Haoyu Wang, Haoyu Xing},
  booktitle={Proceedings of the ...},
  year={2024},
  organization={Tianjin University and Fudan University}
}



