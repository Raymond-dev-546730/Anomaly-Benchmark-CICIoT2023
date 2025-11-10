# Anomaly Detection for Zero-Day IoT Attacks

This repository contains training scripts and terminal logs from:
> **"Performance–Efficiency Trade-offs in Anomaly Detection Architectures for Zero-Day IoT Attack Detection: A Systematic Benchmark on CICIoT2023"**  
> *Accepted to Workshop on AI for Cyber Threat Intelligence (WAITI) 2025*  
> Raymond Lee & Yunpeng Zhang

## Overview

We benchmark **10** anomaly detection models for zero-day IoT intrusion detection across tree-based, kernel, probabilistic, neural, and hybrid paradigms. All models are trained **exclusively on benign traffic** and evaluated on **33 unseen attack types** from the CICIoT2023 dataset containing **46.7 million network flows**. Experiments are conducted on resource-constrained hardware simulating IoT gateway environments, measuring both detection performance and computational efficiency to identify practical deployment trade-offs.

## Repository Contents
~~~
Scripts/
├── Autoencoder_CICIoT2023.py           # Autoencoder training pipeline
├── VAE_CICIoT2023.py                   # Variational Autoencoder training pipeline
├── VQVAE_CICIoT2023.py                 # Vector-Quantized Variational Autoencoder training pipeline
├── DVAE_CICIoT2023.py                  # Denoising Autoencoder training pipeline
├── LSTM_CICIoT2023.py                  # Long Short-Term Memory Autoencoder training pipeline
├── DeepSVDD_CICIoT2023.py              # Deep Support Vector Data Description training pipeline
├── GMM_CICIoT2023.py                   # Gaussian Mixture Model training pipeline
├── SVM_CICIoT2023.py                   # One-Class Support Vector Machine training pipeline
├── IF_CICIoT2023.py                    # Isolation Forest training pipeline
└── EE_CICIoT2023.py                    # Elliptic Envelope training pipeline

Terminal_Logs/
├── Autoencoder Terminal Logs.txt                              # Autoencoder training output
├── Variational Autoencoder Terminal Logs.txt                  # Variational Autoencoder training output
├── Vector-Quantized Variational Autoencoder Terminal Logs.txt # Vector-Quantized Variational Autoencoder training output
├── Denoising Autoencoder Terminal Logs.txt                    # Denoising Autoencoder training output
├── Long Short-Term Memory Autoencoder Terminal Logs.txt       # Long Short-Term Memory Autoencoder training output
├── Deep Support Vector Data Description Terminal Logs.txt     # Deep Support Vector Data Description training output
├── Gaussian Mixture Model Terminal Logs.txt                   # Gaussian Mixture Model training output
├── One-Class Support Vector Machine Terminal Logs.txt         # One-Class Support Vector Machine training output
├── Isolation Forest Terminal Logs.txt                         # Isolation Forest training output
└── Elliptic Envelope Terminal Logs.txt                        # Elliptic Envelope training output
~~~

## Key Results

- **Highest ROC-AUC:** 99.58±0.01% (LSTM Autoencoder)
- **Best Balance:** 99.44±0.03% ROC-AUC with 87s training (Standard Autoencoder)
- **Smallest Model:** 36KB (Autoencoder)
- **Fastest Training:** 0.4s (Isolation Forest, but only 92.96% ROC-AUC)

Standard Autoencoder offers the best performance-efficiency trade-off for resource-constrained deployments.

## Dataset

The CICIoT2023 dataset is available at: 

## Citation

If you use this code or reference our work, please cite:
```bibtex
@inproceedings{lee2025zeroday,
 title={Performance–Efficiency Trade-offs in Anomaly Detection Architectures for Zero-Day IoT Attack Detection: A Systematic Benchmark on CICIoT2023},
 author={Lee, Raymond and Zhang, Yunpeng},
 booktitle={Workshop on AI for Cyber Threat Intelligence (WAITI)},
 year={2025}
}
```
