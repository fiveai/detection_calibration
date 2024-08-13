# On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines

[![arXiv](https://img.shields.io/badge/arXiv-2405.20459-b31b1b.svg)](https://arxiv.org/abs/2405.20459)

The official implementation of ECCV 2024 (Oral🔥) work "On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines".

> [**On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines**](https://arxiv.org/abs/2405.20459)            
> Selim Kuzucu\*, Kemal Oksuz\*, Jonathan Sadeghi, Puneet K. Dokania
> (\* Equal contribution for first authorship)

## Introduction

Building calibrated object detectors is a crucial challenge to address for their reliable usage in safety-critical applications. To enable this functionality, in this repository we include the necessary tools for (i) calibrating object detectors using post-hoc approaches and (ii) evaluating them. Our toolset are designed specifically to respect to the practical usage of the object detectors. 

More particularly, for the evaluation purpose, our benchmark considers the Localization-Recall-Precision (LRP) Error [1] as the primary accuracy measure and Localization-aware ECE (LaECE) [2] as the primary calibration error measure, while also providing support for the well-known AP [3] for measuring accuracy and D-ECE [4] for measuring calibration. As for the calibration purpose, our repository supports (i) Temperature Scaling [5], (ii) Platt Scaling [6], (iii) Linear Regression, (iv) Isotonic Regression [7].

## 1. Installations and Preparations

### Installing the detection_calibration Library

Running the following command will install the library with the relevant requirements:

```
pip install git+https://github.com/fiveai/detection_calibration
```

### Preparing the Datasets

As post-hoc calibration approaches are typically obtained a held-out validation set (which is different from the test set), we randomly split the validation sets of the datasets into two as *minival* and *minitest*. The annotation files for these splits of COCO [3], Cityscapes [8]  and LVIS v1.0 [9] datasets can be downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/1nuv1gr-C8LfkZSRUWaYOwwus_vQ0ZWhM). 

### Additional Preparations

First, create a `data` directory in the root of the directory. This should include the dataset-specific files such as the images and the annotations. To exemplify, for COCO dataset, one would have the following folder structure:

```text
data
├── coco
│   ├── annotations
│   │   ├── calibration_val2017.json
│   │   ├── calibration_test2017.json
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   ├── ...
│   ├── train2017
│   │   ├── 118287 images
│   ├── val2017
│   │   ├── 5000 images
```

Then, create a `detections` directory. This will include the json files of the relevant detectors for both the validation and test sets of the relevant datasets. Note that these detection files **must comply with COCO-style**. To exemplify, for D-DETR on Cityscapes and COCO, one would have the following folder structure:
```text

detections
├── deformable_detr_cityscapes_minival.bbox.json
├── deformable_detr_cityscapes_minitest.bbox.json
├── deformable_detr_coco_minival.bbox.json
├── deformable_detr_coco_minitest.bbox.json
├── ...
```

## 2. Evaluating and Calibrating Object Detectors

### Overview

This repository supports Temperature Scaling [5], Platt Scaling [6], Linear Regression and Isotonic Regression [7] post-hoc calibrators.

### Common Objects, Autonomous Vehicles (and any other!) Benchmarks - Object Detection

```python
from detection_calibration.DetectionCalibration import DetectionCalibration

# Initialize the main calibration class with the validation and test annotation files
calibration_model = DetectionCalibration('data/<dataset_name>/validation.json', 'data/<dataset_name>/test.json')

# Fit the specified calibrator to the validation set
calibrator, thresholds = calibration_model.fit('detections/<model_name>_<dataset_name>_validation.bbox.json', \
    calibrator_type='isotonic_regression')

# Transform the evaluation set with the learned calibrator and the obtained thresholds
cal_test_detections = calibration_model.transform('detections/<model_name>_<dataset_name>_test.bbox.json', \
    calibrator, thresholds)

# Output the final evaluation results for both accuracy and calibration
calibration_model.evaluate_calibration(cal_test_detections)
```

### Long-tailed Object Detection Benchmark (LVIS)

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

Just ensure that the `DetectionCalibration` class is initialized with its `eval_type` argument set to `segm`, and the rest is exactly the same as above examples:

```python
# Initialize the eval_type attribute with 'segm'
calibration_model = DetectionCalibration('data/<dataset_name>/validation.json', 'data/<dataset_name>/test.json', eval_type='segm')
```

### Evaluation Under Domain Shift

It is sufficient to simply pass the relevant testing annotation file with its corresponding detections under COCO-style json files for any given detector. Note that the validation dataset, where the calibrator is learned, should remain the same for proper benchmarking. To exemplify, to evaluate a detector for Common Objects Benchmark under the domain shift dataset Obj45K, the following template can be followed:

```python
from detection_calibration.DetectionCalibration import DetectionCalibration

# Initialize the main calibration class with COCO validation and Obj45K (test) annotation files
calibration_model = DetectionCalibration('data/coco/calibration_val2017.json', 'data/coco/obj45k.json')

# Fit the specified calibrator to the validation set
calibrator, thresholds = calibration_model.fit('detections/<model_name>_coco_validation.bbox.json', \
    calibrator_type='isotonic_regression')

# Transform the Obj45K set with the learned calibrator and the obtained thresholds
cal_test_detections = calibration_model.transform('detections/<model_name>_obj45k.bbox.json', \
    calibrator, thresholds)

# Output the final evaluation results for both accuracy and calibration
calibration_model.evaluate_calibration(cal_test_detections)
```

An analogous case can also be exemplified for evaluating a Cityscapes-trained model under Foggy Cityscapes. This time, however, the test annotation file would remain the same as regular Cityscapes though the COCO-style detection json for the test set would change accordingly:

```python
from detection_calibration.DetectionCalibration import DetectionCalibration

# Initialize the main calibration class with the validation and test annotation files of Cityscapes
calibration_model = DetectionCalibration('data/cityscapes/instancesonly_filtered_gtFine_minival.json', \
    'data/cityscapes/instancesonly_filtered_gtFine_minitest.json.json')

# Fit the specified calibrator to the validation set
calibrator, thresholds = calibration_model.fit('detections/<model_name>_cityscapes_validation.bbox.json', \
    calibrator_type='isotonic_regression')

# Perform transform on the detections of the Foggy Cityscapes dataset with the selected fog density
cal_test_detections = calibration_model.transform('detections/<model_name>_<density>_foggy_cityscapes.bbox.json', \
    calibrator, thresholds)

# Output the final evaluation results for both accuracy and calibration
calibration_model.evaluate_calibration(cal_test_detections)
```
 
## 3. Additional Features

### Evaluating and Calibrating the Detectors For D-ECE

For evaluation in terms of D-ECE after calibration with D-ECE-style targets (binary, based on TP/FP detections):

```python
from detection_calibration.DetectionCalibratoin import DetectionCalibration

# Initialize the main calibration class with is_dece=True, bin_count=10 and IoU TP threshold tau=0.5
calibration_model = DetectionCalibration('data/<dataset_name>/validation.json', 'data/<dataset_name>/', \
    is_dece=True, tau=0.5, bin_count=10)

# Fits the specified calibrator to the validation set, classagnostic with 0.3 thresholds for D-ECE
calibrator, thresholds = calibration_model.fit('detections/<model_name>_<dataset_name>_validation.bbox.json', \
    calibration_type='isotonic_regression', classagnostic=True, threshold_type=[0.3, 0.3])

# Transform the evaluation set with the learned calibrator and the obtained thresholds
cal_test_detections = calibration_model.transform('detections/<model_name>_<dataset_name>_test.bbox.json', \
    calibrator, thresholds)

# Set is_dece=True to observe D-ECE
calibration_model.evaluate_calibration(cal_test_detections, is_dece=True)
```

### Evaluation with Average Precision (AP)

To evaluate the benchmarks with AP and its components in terms of accuracy, simply setting the `verbose=True` while calling the `evaluate_calibration()` is sufficient:

```python
# Set verbose=True to observe AP with its components
calibration_model.evaluate_calibration(cal_test_detections, verbose=True)
```

### Observing Reliability Diagrams

Setting `show_plot` to `True` when calling `evaluate_calibration()` will plot and show the relevant reliability diagram:

```python
# Outputs the final evaluation results for both accuracy and calibration
calibration_model.evaluate_calibration(cal_test_detections, show_plot=True)
```

### Other Features

Please refer to docstrings in the code to see the additional functionality of this repository.

## How to Cite

Please cite the paper if you benefit from our paper or the repository:

```
@misc{kuzucu2024calibration,
      title={On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines}, 
      author={Selim Kuzucu and Kemal Oksuz and Jonathan Sadeghi and Puneet K. Dokania},
      year={2024},
      eprint={2405.20459},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
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
