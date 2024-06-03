from detection_calibration.coco_calibration import CalibrationCOCO
from detection_calibration.lvis_calibration import CalibrationLVIS


class DetectionCalibration:
    def __init__(self, val_annotations, test_annotations, use_lvis=False, eval_type='bbox', bin_count=25, tau=0.0, is_dece=False, is_ace=False, max_dets=100):
        """
        Joint class for learning a post-hoc calibrator, calibrating the
        outputs of a detector and performing joint accuracy/calibration
        benchmarking of any given object detector on any benchmark

        Attributes:
            val_annotations (str): file path for validation set annotations
            test_annotations (str): file path for test set annotations
            eval_type (str): evaluation type, either 'bbox' or 'segm'
            bin_count (int): number of bins to obtain bin-wise calibration errors
            tau (float): IoU threshold for determining TP/FP in evaluation
            is_dece (bool): Whether to use D-ECE-style binary (TP/FP) targets
            is_ace (bool): Whether to perform the evaluation for adaptive CE
            max_dets (int): Max. number of detections per each image to consider
        """

        self.val_annotations = val_annotations
        self.test_annotations = test_annotations

        if use_lvis:
            self.calibration_scheme = CalibrationLVIS(
                val_annotations, test_annotations, eval_type, bin_count, tau, is_dece, is_ace, max_dets=300)
        else:
            self.calibration_scheme = CalibrationCOCO(
                val_annotations, test_annotations, eval_type, bin_count, tau, is_dece, is_ace, max_dets)

    def fit(self, val_detections, calibrator_type, threshold_type=[-1, -1], eval_type='bbox', classagnostic=False, use_grid_search=False, use_quality_focal_loss=False):
        """
        Fit a specified calibrator based on specified thresholds
        :param val_detections  (array array)                 : validation detections
               calibrator_type  (str)                        : fitted calibrator type
               thresholds (float array)                      : pair of thresholds for two stages
               eval_type (str)                               : evaluation type, either 'bbox' or 'segm
               classagnostic (bool)                          : class-agnostic or class-wise calibration
               use_grid_search (bool)                        : grid searching or L-BFGS optimizing TS/PS
               use_quality_focal_loss (bool)                 : objective for optimizing TS/PS
        :return: self.calibration_scheme.fit()               : func. call to the relevant fit()
        """

        return self.calibration_scheme.fit(val_detections, calibrator_type, threshold_type, eval_type, classagnostic, use_grid_search, use_quality_focal_loss)

    def transform(self, test_detections, calibration_model, thresholds):
        """
        Calibrate a given set of detections based on a fitted calibrator
        :param test_detections  (array array)               : test detections
               calibration_model  (calibrator object)       : fitted calibrator type
               thresholds (float array)                     : pair of thresholds for two stages
        :return: self.calibration_scheme.transform()        : func. call to the relevant transform()
        """

        return self.calibration_scheme.transform(test_detections, calibration_model, thresholds)

    def evaluate_calibration(self, calibrated_test_detections, is_dece=False, show_plot=False, verbose=False):
        """
        Joint evaluation of accuracy and calibration
        :param calibrated_test_detections  (array array)         : calibrated test detections
               is_dece  (bool)                                   : print D-ECE or not
               show_plot (bool)                                  : show the reliability diagram or not
               verbose (bool)                                    : print AP components or not
        :return: self.calibration_scheme.evaluate_calibration()  : func. call to the relevant evaluate_calibration()
        """

        return self.calibration_scheme.evaluate_calibration(calibrated_test_detections, is_dece, show_plot, verbose)
