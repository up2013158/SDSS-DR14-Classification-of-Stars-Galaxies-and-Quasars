# SDSS-DR14-Classification-of-Stars-Galaxies-and-Quasars
Classification of Stars, Galaxies and Quasars using Sloan Digital Sky Survey DR14

## Project Description
This project focuses on comparing traditional machine learning (ML) techniques with modern neural network (NN) approaches to solve a classification problem. Using a real-world dataset, we explore how different models perform on the same task and analyze the factors that influence the success of neural network training.

The project is structured to answer three main questions:
1. **Traditional vs. Neural Network Approach:** We apply a traditional machine learning algorithm (e.g., Decision Tree) and a neural network to the dataset, comparing the performance of each method.
2. **Impact of Neural Network Factors:** We investigate how different factors (such as activation functions, data augmentation, and pre-training strategies) affect the performance of the neural network.
3. **Research Exploration:** We dive deeper into one of the research questions concerning neural network optimization, such as how data augmentation or the choice of activation functions influences the model’s effectiveness.

### Key Objectives:
- **Comparison of Algorithms:** Compare a traditional machine learning approach (e.g., Decision Tree) with a neural network to understand the strengths and weaknesses of each approach.
- **Neural Network Performance:** Explore how data handling (augmentation, training data size, class balance) and hyperparameters (activation function, epochs, batch size) affect the performance of a neural network.
- **Research Investigation:** Address a research question related to neural network optimization, providing deeper insights into model improvements.

### Tools and Technologies:
- **Machine Learning:** Scikit-learn for traditional machine learning models.
- **Deep Learning:** TensorFlow/Keras for building neural networks.
- **Data Science:** Pandas and NumPy for data manipulation; Matplotlib for data visualization.
- **Python Dependencies:** A list of dependencies is provided in the `dependencies.txt` file.

The project includes detailed tutorials for beginners (Q1 & Q2) and more advanced discussions for intermediate users (Q3), with a focus on making machine learning and deep learning techniques accessible to a broad audience.

By conducting this project, we aim to provide insights into the strengths and limitations of both traditional and neural network-based approaches. We will not only evaluate the raw performance of each model but also explore the impact of specific factors on neural network training. This will help us and others understand what works best in different machine learning scenarios, especially when considering factors like dataset size, class balance, and computational resources. The goal is to equip the reader with practical knowledge and the tools to make informed decisions when choosing between different machine learning techniques for real-world problems.


## Dataset Overview
This project provides a detailed exploration of astronomical data from the Sloan Digital Sky Survey (SDSS). The dataset can be used for astrophysical research, machine learning, or educational purposes.

This dataset consists of **10,000 records** of observations of space taken by the Sloan Digital Sky Survey (SDSS).  
Each observation is described by **17 feature columns** and **1 target column** that identifies the observation as being a **star**, **galaxy**, or **quasar**.

Further description of the features can be found at:  
[SDSS Glossary](http://skyserver.sdss.org/dr7/en/help/docs/glossary.asp)

To learn more about the SDSS project:  
[SDSS Official Website](http://www.sdss.org/)

### Source
- **Dataset Origin:** The data is from the current data release RD14 of the SDSS.  
- **Access Methods:** There are various ways to access SDSS data. You can find details here:  
  [SDSS Data Access](http://www.sdss.org/dr14/)  

This specific dataset was obtained using a sample query from:  
[SkyServer CasJobs](http://skyserver.sdss.org/casjobs/)

- **Total Rows:** 10,000
- **Total Columns:** 18
- **Data Types:** 
  - 10 columns: `float64`
  - 7 columns: `int64`
  - 1 column: `object` (string)

### Column Descriptions
| Column      | Description                                        | Data Type |
|-------------|----------------------------------------------------|-----------|
| `objid`     | Object identifier                                 | float64   |
| `ra`        | Right ascension                                   | float64   |
| `dec`       | Declination                                       | float64   |
| `u`         | Magnitude in the U filter                         | float64   |
| `g`         | Magnitude in the G filter                         | float64   |
| `r`         | Magnitude in the R filter                         | float64   |
| `i`         | Magnitude in the I filter                         | float64   |
| `z`         | Magnitude in the Z filter                         | float64   |
| `run`       | Run number of the observation                     | int64     |
| `rerun`     | Rerun number for calibration                      | int64     |
| `camcol`    | Camera column number                              | int64     |
| `field`     | Field number in the run                           | int64     |
| `specobjid` | Spectroscopic object identifier                   | float64   |
| `class`     | Classification of the object (e.g., STAR, GALAXY) | object    |
| `redshift`  | Redshift of the object                            | float64   |
| `plate`     | Spectroscopic plate number                        | int64     |
| `mjd`       | Modified Julian Date of the observation           | int64     |
| `fiberid`   | Fiber ID for the spectroscopic observation        | int64     |

### Sample Data
| objid          | ra         | dec        | u       | g       | r       | i       | z       | class  | redshift |
|----------------|------------|------------|---------|---------|---------|---------|---------|--------|----------|
| 1.23765e+18    | 183.531326 | 0.089693   | 19.474  | 17.042  | 15.947  | 15.503  | 15.225  | STAR   | -0.00001 |
| 1.23765e+18    | 183.598370 | 0.135285   | 18.663  | 17.214  | 16.676  | 16.489  | 16.392  | STAR   | -0.00005 |
| 1.23765e+18    | 183.680207 | 0.126185   | 19.383  | 18.192  | 17.474  | 17.087  | 16.801  | GALAXY | 0.12311  |
| 1.23765e+18    | 183.870529 | 0.049911   | 17.765  | 16.603  | 16.161  | 15.982  | 15.904  | STAR   | -0.00011 |
| 1.23765e+18    | 183.883288 | 0.102557   | 17.550  | 16.263  | 16.439  | 16.555  | 16.613  | STAR   | 0.00059  |
| 1.23765e+18    | 184.023450 | 0.073281   | 20.231  | 18.634  | 17.803  | 17.423  | 16.992  | GALAXY | 0.11237  |
| 1.23765e+18    | 184.151283 | 0.120943   | 18.951  | 17.540  | 16.986  | 16.698  | 16.322  | STAR   | -0.00007 |

- The dataset includes magnitudes for celestial objects across multiple filters (`u`, `g`, `r`, `i`, `z`) and metadata for spectroscopic observations.
- Objects are classified as either **STAR** or **GALAXY**, with corresponding redshift values.

## Motivation
The purpose of this project is to explore and compare the performance of traditional machine learning methods against modern neural network approaches in a practical machine learning scenario. By working through various steps and techniques, this project aims to answer key questions about the effectiveness of different algorithms and how various factors influence neural network performance.

### Why Traditional Approaches?
Machine learning has evolved dramatically in recent years, and neural networks have become the go-to solution for many complex problems. However, traditional machine learning algorithms still have their place, especially when dealing with smaller datasets or problems that don’t require the complexity of deep learning models. In this project, we begin by applying a traditional machine learning method to solve a classification problem. The goal here is to establish a baseline for comparison — how well can traditional methods handle the data, and what performance can we expect from a simpler model? We can also examine the ease of implementation, interpretability, and speed of these models, which remain essential factors in certain scenarios.

### Why Neural Networks?
Neural networks have shown outstanding performance in tasks such as image recognition, natural language processing, and more. However, they often come with a steep learning curve and require more data and computational power to achieve optimal performance. The motivation for including a neural network in this project is to explore how well these models perform compared to traditional methods when solving the same classification problem. We are particularly interested in understanding how neural networks can overcome limitations that traditional approaches might struggle with, such as non-linear relationships, complex feature interactions, and scalability. We also investigate the impact of various factors like activation functions, data augmentation, and neural network architecture on model performance.

### The Research Questions
Our project also includes an exploration of key research questions to dive deeper into the nuances of neural network training and performance. These questions aim to understand how different choices — from hyperparameters to data handling — can affect the outcome of a neural network model. For example, data augmentation techniques have become popular in improving the performance of deep learning models, especially when working with limited datasets. We aim to explore how augmenting data can improve neural network performance and generalize to new, unseen data.

Another important area of exploration is the choice of activation functions, which can significantly impact the way a neural network learns and performs. We will experiment with different activation functions to better understand their effects on model training, convergence, and final accuracy.

## Dependencies
- numpy==1.19.2
- pandas==1.1.3
- scikit-learn==0.23.2

## Installation
The repository can be cloned using:

`git clone https://github.com/up2013158/SDSS-DR14-Classification-of-Stars-Galaxies-and-Quasars.git`

Users who want to clone the repository can also install the dependencies using pip:

`!pip install -r dependencies.txt`

This command will install all the libraries listed in the dependencies.txt file, ensuring users have all necessary packages need to run the code.

## Usage

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
