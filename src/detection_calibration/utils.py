import io
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from detection_calibration.platt_scaling import PlattScaling, TemperatureScaling
from detection_calibration.lvis.lvis import LVIS
from detection_calibration.lvis.eval import LVISEval
from detection_calibration.lvis.results import LVISResults
from detection_calibration.pycocotools_lrp.coco import COCO
from detection_calibration.pycocotools_lrp.cocoeval import COCOeval


def threshold_detections(detections, detection_level_threshold, dataset_classes):
    """
    Threshold the detections given the thresholds and dataset clases
    :param detections  (dict array)                     : detections of the model
           detection_level_threshold (float or array)   : confidence thresholds for detections
           dataset_classes (array)                      : classes from the relevant dataset
    :return: detections                                 : thresholded detections
    """

    del_items = []
    for idx, detection in enumerate(detections):
        if detection['score'] < detection_level_threshold[dataset_classes.index(detection['category_id'])]:
            del_items.append(idx)
    for idx in sorted(del_items, reverse=True):
        del detections[idx]
    return detections


def COCO_evaluation(annFile, detections, eval_type='bbox', valid_img=None, remove_img=None, tau=None):
    """
    Generic evaluator for AP, oLRP and more for when verbose=True during printing
    :param annFile (str)              : file path for annotations
           detections  (dict array)   : detections of the model
           eval_type (str)            : evaluation type, either 'bbox' or 'segm'
           valid_img (None or array)  : valid image IDs to evaluate, default keeps all
           remove_img (None or array) : image IDs to leave out of evaluation, default keeps all
           tau (float)                : IoU threshold for determining TP/FP in evaluation
    :return: id_evaluator             : COCOeval object initialized with the current evaluation
    """

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(detections)
    id_evaluator = COCOeval(cocoGt, cocoDt, eval_type)
    if tau:
        id_evaluator.params.iouThrs = np.array([tau])
    if remove_img is not None:
        id_evaluator.params.imgIds = list(
            set(id_evaluator.params.imgIds).difference(remove_img))
    elif valid_img is not None:
        id_evaluator.params.imgIds = list(valid_img)
    id_evaluator.evaluate()
    id_evaluator.accumulate()
    id_evaluator.summarize()
    return id_evaluator


def LVIS_evaluation(annFile, detections, eval_type='bbox', valid_img=None, remove_img=None, tau=None):
    """
    LVIS evaluator for AP, oLRP and more for when verbose=True during printing
    :param annFile (str)              : file path for annotations
           detections  (dict array)   : detections of the model
           eval_type (str)            : evaluation type, either 'bbox' or 'segm'
           valid_img (None or array)  : valid image IDs to evaluate, default keeps all
           remove_img (None or array) : image IDs to leave out of evaluation, default keeps all
           tau (float)                : IoU threshold for determining TP/FP in evaluation
    :return: id_evaluator             : LVISeval object initialized with the current evaluation
    """

    lvisGt = LVIS(annFile)
    lvisDt = LVISResults(lvisGt, detections)
    id_evaluator = LVISEval(lvisGt, lvisDt, eval_type)
    if tau:
        id_evaluator.params.iou_thrs = np.array([tau])
    if remove_img is not None:
        id_evaluator.params.img_ids = list(
            set(id_evaluator.params.img_ids).difference(remove_img))
    elif valid_img is not None:
        id_evaluator.params.img_ids = list(valid_img)
    id_evaluator.evaluate()
    id_evaluator.accumulate()
    id_evaluator.summarize()
    id_evaluator.print_results()
    return id_evaluator


def get_detection_thresholds(annFile, detections, benchmark, thr, tau, eval_type='bbox', max_dets=[100]):
    """
    Get the detection thresholds based on the threshold type, max. detections and tau
    :param annFile (str)              : file path for annotations
           detections  (dict array)   : detections of the model
           benchmark (str)            : benchmark type
           thr (float)                : thresholding type
           tau (float)                : IoU threshold for determining TP/FP in evaluation
           eval_type (str)            : evaluation type, either 'bbox' or 'segm'
           max_dets (int array)       : max. number of detections per each image to consider
    :return: float array              : Obtained class-wise confidence thresholds
    """

    if benchmark == 'lvis':
        lvisGt = LVIS(annFile)
        lvisDt = LVISResults(lvisGt, detections)
        id_evaluator = LVISEval(lvisGt, lvisDt, eval_type)

        id_evaluator.params.area_rng = [id_evaluator.params.area_rng[0]]
        id_evaluator.params.area_rngLbl = ['all']
        id_evaluator.params.iou_thrs = np.array([tau])
        id_evaluator.params.max_dets = max_dets

    else:
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(detections)
        id_evaluator = COCOeval(cocoGt, cocoDt, eval_type)

        id_evaluator.params.areaRng = [id_evaluator.params.areaRng[0]]
        id_evaluator.params.areaRngLbl = ['all']
        id_evaluator.params.iouThrs = np.array([tau])
        id_evaluator.params.maxDets = max_dets

    id_evaluator.evaluate()
    id_evaluator.accumulate()

    # LRP-Optimal Thresholds
    if thr == -1:
        print('Obtaining detection-level threshold using LRP-optimal thresholds...')
        return id_evaluator.eval['lrp_opt_thr'].squeeze()
    else:
        print('Obtaining detection-level threshold using a fixed confidence score...')
        return np.ones(len(id_evaluator.eval['lrp_opt_thr'])) * thr


def load_detections_from_file(path):
    """
    Load the detections given the path to the relevant file
    :param path  (str)                    : path to the detections' json file
    :return: detections (dict array)      : thresholded detections
    """

    f = open(path)
    detections = json.load(f)
    f.close()
    return detections


def identity():
    """
    Trigger identity mapping for the calibrator by returning None
    :return: None
    """

    return None


def isotonic_regression(classagnostic, calibration_info, is_dece=False):
    """
    Fit an isotonic regression post-hoc calibrator to the validation detections
    :param classagnostic (bool)            : class-agnostic or class-wise calibration
           calibration_info (dict)         : encapsulation of all evaluation related info
           is_dece (bool)                  : whether to use D-ECE-style binary (TP/FP) targets
    :return: model (IsotonicRegression)    : fitted IsotonicRegression model
    """

    if classagnostic:
        # For corrupted images, all images with this class is already rejected
        if 'tps' not in calibration_info:
            model = np.zeros(0)
        else:
            # Find total number of valid detections for this class
            valid_dets = np.logical_or(
                calibration_info['tps'], calibration_info['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                model = np.zeros(0)
            else:
                # Note that scores and ious are sorted wrt scores
                valid_scores = calibration_info['scores'][valid_dets].reshape(
                    (-1, 1))
                # Note that scores are already sorted
                if is_dece:
                    valid_labels = calibration_info['tps'][valid_dets]

                else:
                    valid_labels = calibration_info['iou'][valid_dets]

                model = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip').fit(
                    valid_scores, valid_labels)
    else:
        model = dict()
        for cl, cl_input in calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                model[cl] = np.zeros(0)
                continue

            # Find total number of valid detections for this class
            valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                model[cl] = np.zeros(0)
                continue

            # Note that scores and ious are sorted wrt scores
            valid_scores = cl_input['scores'][valid_dets].reshape((-1, 1))
            if is_dece:
                valid_labels = cl_input['tps'][valid_dets]
            else:
                valid_labels = cl_input['iou'][valid_dets]

            model[cl] = IsotonicRegression(
                y_min=0., y_max=1., out_of_bounds='clip').fit(valid_scores, valid_labels)

    return model


def linear_regression(classagnostic, calibration_info, is_dece=False):
    """
    Fit a linear regression post-hoc calibrator to the validation detections
    :param classagnostic (bool)            : class-agnostic or class-wise calibration
           calibration_info (dict)         : encapsulation of all evaluation related info
           is_dece (bool)                  : whether to use D-ECE-style binary (TP/FP) targets
    :return: model (LinearRegression)      : fitted LinearRegression model
    """

    if classagnostic:
        if 'tps' not in calibration_info.keys():
            model = np.zeros(0)
        else:
            # Find total number of valid detections for this class
            valid_dets = np.logical_or(
                calibration_info['tps'], calibration_info['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                model = np.zeros(0)
            else:
                # Note that scores and ious are sorted wrt scores
                valid_scores = calibration_info['scores'][valid_dets].reshape(
                    (-1, 1))
                # Note that scores are already sorted
                if is_dece:
                    valid_labels = calibration_info['tps'][valid_dets]
                else:
                    valid_labels = calibration_info['iou'][valid_dets]

                model = LinearRegression(positive=True).fit(
                    valid_scores, valid_labels)

    else:
        model = dict()
        tps = np.zeros(80)
        all_dets = np.zeros(80)
        for cl, cl_input in calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                model[cl] = np.zeros(0)
                continue

            # Find total number of valid detections for this class
            valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
            tps[cl] = cl_input['tps'].sum()
            num_valid_dets = valid_dets.sum()
            all_dets[cl] = num_valid_dets

            # If no detection, then ignore
            if num_valid_dets == 0:
                model[cl] = np.zeros(0)
                continue

            # Note that scores and ious are sorted wrt scores
            valid_scores = cl_input['scores'][valid_dets].reshape((-1, 1))
            # Note that scores are already sorted
            if is_dece:
                valid_labels = cl_input['tps'][valid_dets]
            else:
                valid_labels = cl_input['iou'][valid_dets]

            model[cl] = LinearRegression(positive=True).fit(
                valid_scores, valid_labels)

        return model


def platt_scaling(classagnostic, calibration_info, is_dece=False, use_grid_search=False, use_quality_focal_loss=False):
    """
    Fit a platt scaling post-hoc calibrator to the validation detections
    :param classagnostic (bool)            : class-agnostic or class-wise calibration
           calibration_info (dict)         : encapsulation of all evaluation related info
           is_dece (bool)                  : whether to use D-ECE-style binary (TP/FP) targets
           use_grid_search (bool)          : grid searching or L-BFGS optimizing TS/PS
           use_quality_focal_loss (bool)   : objective for optimizing TS/PS
    :return: model (PlattScaling)          : fitted PlattScaling model
    """

    if classagnostic:
        # For corrupted images, all images with this class is already rejected
        if 'tps' not in calibration_info.keys():
            model = np.zeros(0)
        else:
            # Find total number of valid detections for this class
            valid_dets = np.logical_or(
                calibration_info['tps'], calibration_info['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                model = np.zeros(0)
            else:
                # Note that scores and ious are sorted wrt scores
                valid_scores = calibration_info['scores'][valid_dets].reshape(
                    (-1, 1))
                # Note that scores are already sorted
                if is_dece:
                    valid_labels = calibration_info['tps'][valid_dets]
                    valid_labels = [
                        1.0 if label else 0.0 for label in valid_labels]

                else:
                    valid_labels = calibration_info['iou'][valid_dets]

                valid_scores, valid_labels = np.array(
                    valid_scores), np.array(valid_labels)
                model = PlattScaling()
                model.fit(valid_scores, valid_labels,
                          use_grid_search, use_quality_focal_loss)
    else:
        model = dict()
        for cl, cl_input in calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                model[cl] = np.zeros(0)
                continue

            # Find total number of valid detections for this class
            valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                model[cl] = np.zeros(0)
                continue

            # Note that scores and ious are sorted wrt scores
            valid_scores = cl_input['scores'][valid_dets].reshape((-1, 1))
            if is_dece:
                valid_labels = cl_input['tps'][valid_dets]
                valid_labels = [
                    1.0 if label else 0.0 for label in valid_labels]

            else:
                valid_labels = cl_input['iou'][valid_dets]

            model[cl] = PlattScaling()
            model[cl].fit(valid_scores, valid_labels,
                          use_grid_search, use_quality_focal_loss)

    return model


def temperature_scaling(classagnostic, calibration_info, is_dece=False, use_grid_search=False, use_quality_focal_loss=False):
    """
    Fit a temperature scaling post-hoc calibrator to the validation detections
    :param classagnostic (bool)                 : class-agnostic or class-wise calibration
           calibration_info (dict)              : encapsulation of all evaluation related info
           is_dece (bool)                       : whether to use D-ECE-style binary (TP/FP) targets
           use_grid_search (bool)               : grid searching or L-BFGS optimizing TS/PS
           use_quality_focal_loss (bool)        : objective for optimizing TS/PS
    :return: model (TemperatureScaling)         : fitted TemperatureScaling model
    """

    if classagnostic:
        # For corrupted images, all images with this class is already rejected
        if 'tps' not in calibration_info.keys():
            model = np.zeros(0)
        else:
            # Find total number of valid detections for this class
            valid_dets = np.logical_or(
                calibration_info['tps'], calibration_info['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                model = np.zeros(0)
            else:
                # Note that scores and ious are sorted wrt scores
                valid_scores = calibration_info['scores'][valid_dets].reshape(
                    (-1, 1))
                # Note that scores are already sorted
                if is_dece:
                    valid_labels = calibration_info['tps'][valid_dets]
                    valid_labels = [
                        1.0 if label else 0.0 for label in valid_labels]

                else:
                    valid_labels = calibration_info['iou'][valid_dets]

                valid_scores, valid_labels = np.array(
                    valid_scores), np.array(valid_labels)
                model = TemperatureScaling()
                model.fit(valid_scores, valid_labels,
                          use_grid_search, use_quality_focal_loss)
    else:
        model = dict()
        for cl, cl_input in calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                model[cl] = np.zeros(0)
                continue

            # Find total number of valid detections for this class
            valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                model[cl] = np.zeros(0)
                continue

            # Note that scores and ious are sorted wrt scores
            valid_scores = cl_input['scores'][valid_dets].reshape((-1, 1))
            if is_dece:
                valid_labels = cl_input['tps'][valid_dets]
                valid_labels = [
                    1.0 if label else 0.0 for label in valid_labels]

            else:
                valid_labels = cl_input['iou'][valid_dets]

            model[cl] = TemperatureScaling()
            model[cl].fit(valid_scores, valid_labels,
                          use_grid_search, use_quality_focal_loss)

    return model
