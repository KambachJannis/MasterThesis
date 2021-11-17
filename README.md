# Semantic Segmentation of Satellite Images Using Point Supervision

Master Thesis by Jannis Kambach, WWU Münster

## Motivation

Considering the steadily growing volume of available satellite images and the increasing importance of deep learning for analyzing them on a larger scale, the main goal of this work is to evaluate weakly superised learning as an approach to reduce the dependence on manually created labels. As shown by [Bearman et al.](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_34), a segmentation model trained on point-lables can outperform a fully supervised model given the same annotation time budget.

## Summary of the Main Findings
- Most methods rely on pseudo-masks to produce accurate segmentation results - generalized methods that measure the likelihood that an image region contains an object
- Early comparison tests of different objectness methods show poor performance on satellite images compared to images from benchmark sets such as PASCAL VOC
- [Laradji et al.](https://openaccess.thecvf.com/content_ECCV_2018/html/Issam_Hadj_Laradji_Where_are_the_ECCV_2018_paper.html) propose a custom loss function constructed around the watershed-algorithm that enables full segmentation without pseudo-masks and objectness methods
- Their basic method falls short of the fully supervisied model given the same annotation time budget
- Integrating the [COB objectness method](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_35) into the custom loss function improves segmentation performance
- Training a U-Net on the noisy segmentation maps produced by the improved model improves segmentation performance again

## Implementation

- 0_Objectness:
- 1_Preprocessing:
- 2_Training:
- 3_Testing:

## Usage

The preprocessing script expects a dataset of satellite images consisting of 1km x 1km GeoTIFF files alongside a shapefile for the labels.
The main contributions are the custom loss functions in the *losses* folder.
