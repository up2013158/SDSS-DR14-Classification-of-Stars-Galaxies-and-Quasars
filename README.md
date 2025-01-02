# SDSS-DR14-Classification-of-Stars-Galaxies-and-Quasars
Classification of Stars, Galaxies and Quasars using Sloan Digital Sky Survey DR14

## Description
Classification of Stars, Galaxies, and Quasars using Sloan Digital Sky Survey DR14.

## Installation

## Usage

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

# Motivation
Describe the questions or motivations behind the project.

# Dependencies
- numpy==1.19.2
- pandas==1.1.3
- scikit-learn==0.23.2

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Contact
