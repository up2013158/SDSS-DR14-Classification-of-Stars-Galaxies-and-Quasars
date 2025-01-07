# SDSS-DR14-Classification-of-Stars-Galaxies-and-Quasars
Classification of Stars, Galaxies and Quasars using Sloan Digital Sky Survey DR14

## Project Description
This project focuses on comparing traditional machine learning (ML) techniques with modern neural network (NN) approaches to solve a classification problem. Using a real-world dataset from SDSS, we explore how different the models perform and analyze the factors that influence the success of neural network training. As raw optical images of the stars, galaxies and quasars in the survey can visably appear similar as 'point sources' on the sky, it can prove diffcult to calssify the type of object being observed. Therefore, the testing and implementation of ML and NN techniques on a dataset like SDSS DR14 can a provide quick and effective method for object classification. The results from the ML and NN models can be compared to the `class` column in the dataset to evaulte the accuracy and limitations of each technique.

The project is structured to answer three main questions:
1. **Traditional vs. Neural Network Approach:** We apply a traditional machine learning algorithm (Decision Tree) and a neural network to the dataset, comparing the performance of each method.
2. **Impact of Neural Network Factors:** We investigate how different factors (such as activation functions, data augmentation, and pre-training strategies) affect the performance of the neural network.
3. **Research Exploration:** We dive deeper into one of the research questions concerning neural network optimization, how do choices about data such as amount of training data and balance of classes in a classification problem affect the performance of a neural network?

### Key Objectives:
- **Comparison of Algorithms:** Compare a traditional machine learning approach (Decision Tree) with a neural network to understand the strengths and weaknesses of each approach.
- **Neural Network Performance:** Explore how data handling (augmentation, training data size, class balance) and hyperparameters (activation function, epochs, batch size) affect the performance of a neural network.
- **Research Investigation:** Address a research question related to neural network optimization, providing deeper insights into model improvements.

### Tools and Technologies:
- **Machine Learning:** Scikit-learn for traditional machine learning models.
- **Deep Learning:** TensorFlow/Keras for building neural networks.
- **Data Science:** Pandas and NumPy for data manipulation; Matplotlib for data visualization.
- **Python Dependencies:** A list of dependencies is provided in the `dependencies.txt` file.

The project includes detailed tutorials for beginners (Q1 & Q2) and more advanced discussions for intermediate users (Q3), with a focus on addressing the reshearch quesion for NN optimization.

By conducting this project, we aim to provide insights into the strengths and limitations of both traditional and neural network-based approaches. We will not only evaluate the raw performance of each model but also explore the impact of specific factors on neural network training. This will help us and others understand what works best in different machine learning scenarios, especially when considering factors like dataset size, class balance, and computational resources. The goal is to equip the reader with practical knowledge and the tools to make informed decisions when choosing between different machine learning techniques for real-world problems.

## Dataset Overview
This project provides a detailed exploration of astronomical data from the Sloan Digital Sky Survey (SDSS) Data Release 14 (DR14). The dataset consists of **10,000** observations of taken from the fourth phase of the Sloan Digital Sky Survey (SDSS-IV). Each observation is described by **17 feature columns** and **1 target column** that identifies the observation as being a **star**, **galaxy**, or **quasar**. The catalog includes object position in the sky (`ra` and `dec`), redshift (`z`) and the observed apparent magnitude in each filter band, each are crital parameters to training the ML or NN classifiers. 

Further details of the data set features can be found at:  
[SDSS Glossary](http://skyserver.sdss.org/dr7/en/help/docs/glossary.asp)

To learn more about SDSS:  
[SDSS Official Website](http://www.sdss.org/)

### Source
- **Dataset Origin:** The data is from the data release DR14 of the SDSS.  
- **Data Access:** There are various ways to gain access to the SDSS data, however more details can be found here:  
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

- The dataset includes observed magnitudes for the objects across multiple filter bands (`u`, `g`, `r`, `i`, `z`) and metadata for spectroscopic observations with corresponding redshift values.
- Objects are also classified as either **STAR**, **GALAXY** or **QUASAR** which is the classification problem we tested our ML and NN models against.

## Motivation
The purpose of this project is to explore and compare the performance of traditional machine learning methods against modern neural network approaches in a practical machine learning scenario. By working through various steps and techniques, this project aims to answer key questions about the effectiveness of different algorithms and how various factors influence neural network performance.

### Why Traditional Approaches?
Machine learning has evolved dramatically in recent years, and neural networks have become the go-to solution for many complex problems. However, traditional machine learning algorithms still have their place, especially when dealing with smaller datasets or problems that don’t require the complexity of deep learning models. In this project, we begin by applying a simple ('easy-to-implement') traditional machine learning method (desicion tree) to solve a classification problem. The goal is to establish how well can the traditional method handles the data, and what kind of performance can be expected from a simple model. We also examine the ease of implementation, interpretability, and speed of these models, which remain essential factors in certain scenarios.

### Why Neural Networks?
Neural networks today are critical in AI, image recognition and language processing. However, they are often require much more expertise to implement as well more training data and computational power to achieve optimal performance. The motivation for including a neural network in this project is to explore how well these models perform compared to traditional methods when solving the same classification problem. We are particularly interested in understanding how neural networks can overcome limitations that traditional approaches might struggle with, such as non-linear relationships, complex feature interactions, and scalability. We also investigate the impact of various factors like activation functions, data augmentation, and neural network architecture on model performance.

### The Research Questions
Our project also includes an exploration of key research questions to dive deeper into the nuances of neural network training and performance. These questions aim to understand how different choices — from hyperparameters to data handling — can affect the outcome of a neural network model. For example, data augmentation techniques have become popular in improving the performance of deep learning models, especially when working with limited datasets. We aim to explore how augmenting data can improve neural network performance and generalize to new, unseen data.

Another important area of exploration is the choice of activation functions, which can significantly impact the way a neural network learns and performs. We will experiment with different activation functions to better understand their effects on model training, convergence, and final accuracy.

## Dependencies
- numpy==1.19.2
- pandas==1.1.3
- scikit-learn==0.23.2

## Installation and Usage
Users who want to install the dependencies using pip:

`!pip install -r dependencies.txt`

This command will install all the libraries listed in the dependencies.txt file, ensuring users have all necessary packages need to run the code.

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

## References
Acknowledgment for ChatGPT AI:
- Tool Name: ChatGPT
- Provider: OpenAI
- Description: ChatGPT was used to assist in brainstorming, debugging, and generating portions of the codebase for this project.
- Accessed via: https://openai.com/chatgpt

Acknowledgment for Gemini AI:
- Tool Name: Gemini AI
- Provider: Google DeepMind
- Description: Gemini AI was utilized to generate suggestions, optimize algorithms, and enhance the project's overall design.
- Accessed via: Google Colab notebook integration.

Acknowledgment for Claude AI:
- Tool Name: Claude
- Provider: Anthropic
- Description: Claude was employed to generate, debug, and refine code for this project.
- Accessed via: Visual Studio Code extension for Claude AI.

## Contact
Samuel Helps

samuel.helps.sh@gmail.com

Physics, Astrophysics and Cosmology (Mphys)

University of Portsmouth
