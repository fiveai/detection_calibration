# On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines

[![arXiv](https://img.shields.io/badge/arXiv-2405.20459-b31b1b.svg)](https://arxiv.org/abs/2405.20459)

The official implementation of "On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines". This work is accepted to **ECCV 2024 for an oral presentation**.

> [**On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines**](https://arxiv.org/abs/2405.20459)            
> Selim Kuzucu\*, Kemal Oksuz\*, Jonathan Sadeghi, Puneet K. Dokania
> (\* Equal contribution for first authorship)


## How to Cite

Please cite the paper if you benefit from our paper or the repository:

```
@inproceedings{kuzucu2024calibration,
      title={On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines}, 
      author={Selim Kuzucu and Kemal Oksuz and Jonathan Sadeghi and Puneet K. Dokania},
      booktitle = {The European Conference on Computer Vision (ECCV)},
      year = {2024}
}
```

## Introduction

Building calibrated object detectors is a crucial challenge to address for their reliable usage in safety-critical applications. To enable this functionality, in this repository we include the necessary tools for (i) calibrating object detectors using post-hoc approaches and (ii) evaluating them. Our toolset are designed specifically to respect to the practical usage of the object detectors. 

More particularly, for the evaluation purpose, our benchmark considers the Localization-Recall-Precision (LRP) Error [1] as the primary accuracy measure and Localization-aware ECE (LaECE) [2] as the primary calibration error measure, while also providing support for the well-known AP [3] for measuring accuracy and D-ECE [4] for measuring calibration. As for the calibration purpose, our repository supports (i) Temperature Scaling [5], (ii) Platt Scaling [6], (iii) Linear Regression, (iv) Isotonic Regression [7].

## Installations and Preparations

### Installing the detection_calibration Library

You can pip install the repository and use it as a dependency:
```
pip install git+https://github.com/fiveai/detection_calibration
```

Or alternatively, you can simply clone the repository and install the dependencies in the requirements first then run,
```
python setup.py develop
```
to use the repository in the development mode.

### Preparing the Repository and the 

As post-hoc calibration approaches are typically obtained a held-out validation set (which is different from the test set), we randomly split the validation sets of the datasets into two as *minival* and *minitest*. The annotation files (as well as some example detection files used in the tutorials) for these splits of COCO [3], Cityscapes [8]  and LVIS v1.0 [9] datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1cl-rHUOCsrkL8KyD_dPpgc4OHU2P9FFO/view?usp=sharing). Please unzip this zip file and place it under the root of this directory if you cloned this directory. If you installed this directory using pip, then please use our tutorials by setting the paths accordingly.

## Tutorials to Calibrate and Evaluate Object Detectors

We provide two tutorials under `tutorials` directory:
- autonomous_driving.py
- common_objects.py

Common objects tutorial is more comprehensive and provide you different functionalities of our repository (the evaluation we propose, evaluation under domain shift, Detection-ECE style evaluation and the standard Average Precision). These tutorials will help you to reproduce some of our results in our paper. For both of the tutorials, we use detections from Deformable-DETR, which is also the main detector we employ in our paper. 

We provide further details about the functionalities that are not covered in these tutorials in the following.
### Long-tailed Object Detection Benchmark (LVIS)
We also have a plan to release a tutorial for the LVIS dataset. Before we release it, if you want to calibrate your models on LVIS dataset, please see the example below:

```python
from detection_calibration.DetectionCalibration import DetectionCalibration

# Initialize the main calibration class by setting use_lvis=True
calibration_model = DetectionCalibration('data/lvis_v1/annotations/lvis_v1_minival.json', \
    'data/lvis_v1/annotations/lvis_v1_minitest.json', use_lvis=True)

# Fit the specified calibrator to the validation set
calibrator, thresholds = calibration_model.fit('detections/<model_name>_lvis_v1_minival_detection.bbox.json', \
    calibrator_type='isotonic_regression')

# Transform the evaluation set with the learned calibrator and the obtained thresholds
cal_test_detections = calibration_model.transform('detections/<model_name>_lvis_v1_minitest_detection.bbox.json', \
    calibrator, thresholds)

# Output the final evaluation results for both accuracy and calibration
calibration_model.evaluate_calibration(cal_test_detections)
```
### Instance Segmentation

We also support instance segmentation in addition to object detection. To evaluate an instance segmentation approach, you can follow our common objects tutorial by ensuring that `DetectionCalibration` is initialized with  `eval_type=segm`:

```python
# Initialize the eval_type attribute with 'segm'
calibration_model = DetectionCalibration('data/<dataset_name>/validation.json', 'data/<dataset_name>/test.json', eval_type='segm')
```
 
### References
- [[1](https://arxiv.org/pdf/2011.10772.pdf)] One Metric to Measure them All: Localisation Recall Precision (LRP) for Evaluating Visual Detection Tasks, TPAMI in 2022 and ECCV 2018  
- [[2](https://arxiv.org/abs/2307.00934)] Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration, CVPR 2023  
- [[3](https://arxiv.org/abs/1405.0312)] Microsoft COCO: Common Objects in Context, ECCV 2014
- [[4](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf)] Multivariate Confidence Calibration for Object Detection, CVPR-W 2020
- [[5](https://arxiv.org/abs/1706.04599)] On Calibration of Modern Neural Networks, ICML 2017
- [6] Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods, Advances in Large Margin Classifiers in 1999
- [7] Transforming classifier scores into accurate multiclass probability estimates, SIGKDD 2002
- [[8](https://arxiv.org/pdf/1604.01685)] The Cityscapes Dataset for Semantic Urban Scene Understanding, CVPR 2016
- [[9](https://arxiv.org/abs/1908.03195)] LVIS: A Dataset for Large Vocabulary Instance Segmentation, CVPR 2019
