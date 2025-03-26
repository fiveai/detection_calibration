from detection_calibration.DetectionCalibration import DetectionCalibration


##### 1. The Evaluation Proposed in Our Paper #####

# Initialize the calibration class with the validation and test annotation files
calibration_model = DetectionCalibration('calibration/data/coco/calibration_val2017.json', 'calibration/data/coco/calibration_test2017.json')

# Fit the specified calibrator to the validation set by providing the path to the detections in coco format on val set.
# calibrator_type can be ['isotonic_regression', 'linear_regression','platt_scaling', 'temperature_scaling', 'identity']
# 'identity' corresponds to uncalibrated detections.
calibrator, thresholds = calibration_model.fit('calibration/detections/deformable_detr_coco_minival.bbox.json', calibrator_type='isotonic_regression')
# Returned Params: 
# - calibrator is the learned calibrator. By default, it returns a calibrator for each class in the dataset.
# - thresholds corresponds to LRP optimal thresholds before calibration (to determine the set of detections used to train the
# calibrator) and after calibration (to keep the detections that survives background removal threshold). 
# Both thresholds are cross-validated using LRP Error on val set in a class-specific manner by default. 
# That is why, each threshold is an C-dim array where C is the number of classes.

# More details on calibration_model.fit
# The following arguments can be used to obtain customized calibrators
# - threshold_type=[-1, -1] is the default, and will validate LRP-optimal threshold. You can also fix each threshold within [0, 1] 
# - eval_type='bbox' by default. For segmentation task one can use 'segm'.
# - classagnostic=False by default, For a single calibrator for all classes, you can set it True
# - use_grid_search=False and use_quality_focal_loss=False are the optimizer settings for temperature scaling and platt scaling.
# You can do grid search without any optimizer, or use quality focal instead of cross entropy as a target as IoUs are between [0,1]
# while optimizing the parameters of these calibrators.

# Transform the confidence scores of the detections on the test set by 
# - first filtering out the detections before calibration, 
# - then applying the calibrator,
# - finally using the second threshold after calibration to return the final set of detections.
cal_test_detections = calibration_model.transform('calibration/detections/deformable_detr_coco_minitest.bbox.json', calibrator, thresholds)

# Print out accuracy and calibration evaluation results
calibration_model.evaluate_calibration(cal_test_detections)

# You can also plot reliability diagram while evaluating the model
# calibration_model.evaluate_calibration(cal_test_detections, show_plot=True)


##### 2. Evaluating the models Under Domain Shift #####

# If you want to evaluate the model on common corruptions as domain shifted dataset,
# you can simply continue calling the calibration_model.transform and evaluate_calibration.
# Currently we do not provide these detection files under domain shift.

# cal_test_detections_fog_1 = calibration_model.transform('calibration/detections/deformable_detr_coco_minitest_fog_1.bbox.json', calibrator, thresholds)
# calibration_model.evaluate_calibration(cal_test_detections_fog_1)


##### 3. D-ECE Style Evaluation #####
# D-ECE style evaluation is different from our evaluation. Please refer to our paper for a comparison between them.
# Below we provide, the code for producing results with this type of evaluation as well.

# Initialize the main calibration class with is_dece=True, bin_count=10 and IoU TP threshold tau=0.5
calibration_model_dece = DetectionCalibration('calibration/data/coco/calibration_val2017.json', 'calibration/data/coco/calibration_test2017.json', \
    is_dece=True, tau=0.5, bin_count=10)

# Fits the specified calibrator to the validation set, classagnostic with 0.3 thresholds for D-ECE
calibrator, thresholds = calibration_model_dece.fit('calibration/detections/deformable_detr_coco_minival.bbox.json', \
    calibrator_type='isotonic_regression', classagnostic=True, threshold_type=[0.3, 0.3])

# Transform the evaluation set with the learned calibrator and the obtained thresholds
cal_test_detections = calibration_model_dece.transform('calibration/detections/deformable_detr_coco_minitest.bbox.json', calibrator, thresholds)

# Set is_dece=True to see D-ECE
calibration_model_dece.evaluate_calibration(cal_test_detections, is_dece=True)

# Initialize the main calibration class with is_dece=True, bin_count=10 and IoU TP threshold tau=0.5
calibration_model_dece = DetectionCalibration('calibration/data/coco/calibration_val2017.json', 'calibration/data/coco/calibration_test2017.json', \
    is_dece=True, tau=0.5, bin_count=10)

##### 4. Getting the Average Precision #####
# You can also print out AP following the standard practice.

# We do not fit any calibrator by default while computing the AP. However, as our provided calibrators are monotonically-increasing
# functions, they preserve the ranking of the detections. That is why, they do not have an effect on the resulting 
# AP (or can have minimal effect as they are not strictly monotonically increasing).
# The threshold needs to be set as 0, otherwise AP will decrease with higher thresholds as it is an area under the curve measure.
calibrator, thresholds = calibration_model_dece.fit('calibration/detections/deformable_detr_coco_minival.bbox.json', \
    calibrator_type='identity', classagnostic=True, threshold_type=[0., 0.])

# Transform the evaluation set with the learned calibrator and the obtained thresholds
cal_test_detections = calibration_model_dece.transform('calibration/detections/deformable_detr_coco_minitest.bbox.json', calibrator, thresholds)

# Set verbose=True to observe AP with its components
calibration_model_dece.evaluate_calibration(cal_test_detections, verbose=True)