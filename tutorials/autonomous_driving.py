from detection_calibration.DetectionCalibration import DetectionCalibration

# Initialize the calibration class with the validation and test annotation files
calibration_model = DetectionCalibration('calibration/data/cityscapes/instancesonly_filtered_gtFine_minival.json',\
                                         'calibration/data/cityscapes/instancesonly_filtered_gtFine_minitest.json')

# Fit the specified calibrator to the validation set
# calibrator_type can be ['isotonic_regression', 'linear_regression','platt_scaling', 'temperature_scaling', 'identity']
# 'identity' corresponds to uncalibrated detections.
calibrator, thresholds = calibration_model.fit('calibration/detections/deformable_detr_cityscapes_minival.bbox.json', calibrator_type='isotonic_regression')
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

# Transform the confidence scores of the detections on the test set by 
# - first filtering out the detections before calibration, 
# - then applying the calibrator,
# - finally using the second threshold after calibration to return the final set of detections.
cal_test_detections = calibration_model.transform('calibration/detections/deformable_detr_cityscapes_minitest.bbox.json', calibrator, thresholds)

# Print out accuracy and calibration evaluation results
calibration_model.evaluate_calibration(cal_test_detections)

# If you want to evaluate the model on foggy cityscapes as domain shifted dataset,
# you can simply continue calling the calibration_model.transform and evaluate_calibration.
# Currently we do not provide these detection files under domain shift.

# cal_test_detections_foggy = calibration_model.transform('calibration/detections/deformable_detr_cityscapes_minitest_foggy.bbox.json', calibrator, thresholds)
# calibration_model.evaluate_calibration(cal_test_detections_fog_1)

# Please refer to common_objects tutorial for further details on the repo.