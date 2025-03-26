from detection_calibration.DetectionCalibration import DetectionCalibration

# Initialize the calibration class with the validation and test annotation files
calibration_model = DetectionCalibration('data/cityscapes/instancesonly_filtered_gtFine_minival.json', 'data/cityscapes/instancesonly_filtered_gtFine_minitest.json')

# Fit the specified calibrator to the validation set
# calibrator_type can be ['isotonic_regression', 'linear_regression','platt_scaling', 'temperature_scaling', 'identity']
# 'identity' corresponds to uncalibrated detections.
calibrator, thresholds = calibration_model.fit('detections/deformable_detr_cityscapes_minival.bbox.json', calibrator_type='isotonic_regression')
# Returned Params: 
# - calibrator is the learned calibrator,
# - thresholds corresponds to LRP optimal thresholds before calibration (to determine the set of detections used to train the
# calibrator) and after calibration (to keep the detections that survives background removal threshold). 
# Both thresholds are cross-validated using LRP Error on val set in a class-specific manner by default. 
# That is why, each threshold is an C-dim array where C is the number of classes.

# Transform the evaluation set by filtering out the detections before calibration, then applying the calibrator,\
# and finally apply the second threhold after calibration to return the final set of detections.
cal_test_detections = calibration_model.transform('detections/deformable_detr_cityscapes_minitest.bbox.json', calibrator, thresholds)

# Print out accuracy and calibration evaluation results
calibration_model.evaluate_calibration(cal_test_detections)