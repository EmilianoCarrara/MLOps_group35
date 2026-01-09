# MLOps_group35

Project synopsis - Group - 35

Title: Identifying neurodiversity 

## Motivation and Goal

Traditional diagnostic approaches for identifying Attention Deficit Hyperactivity Disorder (ADHD) primarily rely on symptom-based criteria. However, growing evidence suggests that criteria-based identification of ADHD may not fully capture the underlying neurobiological diversity associated with ADHD. Moreover, ADHD is a highly heterogeneous neurodevelopmental condition, meaning that there are multiple subgroups within ADHD itself. Therefore, it is important not only to determine whether an individual has ADHD, but also to identify which neurobiological subgroup they belong to based on Structural Magnetic Resonance Imaging (sMRI).

## Scope and Expected Models

This project aims to identify neurodiversity using unsupervised machine learning techniques within a reproducible ML pipeline. The scope is limited to the analysis of structural brain images, and does not include functional MRI, behavioral assessment, or treatment-related outcomes. The project involves selecting the appropriate features that the dataset offers, followed by dimensionality reduction, clustering to identify the different groups, and analysing the clusters produced by the clustering algorithms. First of all we will use the python library called "nibabel" to read 3D imaging files in NIfTI (.nii / .nii.gz) format, then we will proceed to the dimensionality reduction thanks to the use of PCA. After that, we plan to use K-means clustering algorithm, but we would also like to try out a few more clustering algorithms. Finally, we will proceed with an in-depth analysis of the associated metadata to find any significant patterns and correlations within the identified clusters.

## Dataset

The project uses the ADHD200 Preprocessed Anatomical Dataset (anonymized and publicly available for research purposes), which consists of:
- Preprocessed 3D structural MRI scans (sMRI) in NIfTI format.
- Associated phenotypic information such as age, sex, and diagnostic label.
- Data collected from multiple imaging sites, reflecting real-world heterogeneity.

Due to the large size of the dataset (exceeding 20 GB) and the presence of inter-site variability arising from differences in MRI scanners, acquisition protocols, and image quality, the initial analysis will focus on data from a subset of acquisition sites (specifically NYU, OHSU and Peking_1). This choice helps reduce computational complexity and limit differences between data coming from different acquisition sites, while still allowing us to extend the analysis to additional sites later in the project.

## References: 

https://www.kaggle.com/datasets/purnimakumarrr/adhd200-preprocessed-anatomical-dataset
https://fcon_1000.projects.nitrc.org/indi/adhd200/



