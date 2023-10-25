# Anomaly Detection in Network Traffic

## Project Overview
This project is dedicated to the development of an anomaly detection system employing a hybrid approach that seamlessly blends K-Means clustering with Autoencoder neural networks. The primary objective of this model is to discern anomalies within intricate datasets where conventional techniques may exhibit limitations.

## Project Structure
### Data Preprocessing
The initial stages encompass crucial data preprocessing steps, such as dimensionality reduction using Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE), data cleansing, and rigorous feature scaling.

### Model Architecture
The project features a hybrid model integrating K-Means clustering with Autoencoders to enable precise anomaly detection. K-Means clustering is employed for data grouping, while Autoencoders facilitate data reconstruction to pinpoint anomalies.

### Hyperparameter Tuning
Fine-tuning of hyperparameters plays a pivotal role in achieving optimal performance of the hybrid model. These parameters include the number of clusters and encoding dimensions.

### Anomaly Detection
The model adeptly identifies anomalies by contrasting the reconstruction error with a predetermined threshold. Anomalies, characterized by elevated reconstruction errors, are successfully pinpointed.

### Evaluation Metrics
Evaluation metrics, including the Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index, are utilized to assess the model's efficacy when working with unlabelled data. Furthermore, dynamic thresholding is employed for anomaly classification.

### Threshold Selection
The project delves into the meticulous process of dynamic threshold selection, an essential step in segregating anomalies based on their reconstruction errors.

### Visual Inspection
To gain comprehensive insights into the dataset's anomalies, the project incorporates data visualization. This includes the use of scatter plots and heatmaps for a detailed understanding of the anomalies.

### Addressing Class Imbalance
The project discusses techniques for tackling class imbalance, emphasizing the oversampling of anomalies as an effective strategy to enhance the model's overall performance.

### Model Finalization
The final model is arrived at by striking a judicious balance between precision and recall. This critical decision ensures that the model effectively accomplishes its objective.

## Getting Started
- Download the dataset from https://data.mendeley.com/datasets/ztyk4h3v6s/2
- Execute the provided Jupyter notebooks or scripts.

## Requirements
- Python (Version 3.12)
- Essential libraries, including:
  - NumPy
  - TensorFlow
  - Keras
  - Scikit-learn
  - Matplotlib
  - Pandas
  - SciPy
  - t-SNE
  - Other Python libraries for data manipulation and general-purpose tasks.
