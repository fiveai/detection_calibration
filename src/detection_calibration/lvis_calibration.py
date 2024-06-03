import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from detection_calibration.lvis.lvis import LVIS
from detection_calibration.lvis.eval import LVISEval
from detection_calibration.lvis.results import LVISResults

from detection_calibration.utils import threshold_detections, LVIS_evaluation, get_detection_thresholds, load_detections_from_file, identity, isotonic_regression, linear_regression, platt_scaling, temperature_scaling


class CalibrationLVIS(LVISEval):
    def __init__(self, val_annotations, test_annotations, eval_type='bbox', bin_count=25, tau=0.0, is_dece=False, is_ace=False, max_dets=300):
        """
        Class for learning a post-hoc calibrator, calibrating the
        outputs of a detector and performing joint accuracy/calibration
        benchmarking of any given object detector on LVIS benchmarks

        Arguments:
            val_annotations (str)   : file path for validation set annotations
            test_annotations (str)  : file path for test set annotations
            eval_type (str)         : evaluation type, either 'bbox' or 'segm'
            bin_count (int)         : number of bins to obtain bin-wise calibration errors
            tau (float)             : IoU threshold for determining TP/FP in evaluation
            is_dece (bool)          : whether to use D-ECE-style binary (TP/FP) targets
            is_ace (bool)           : whether to perform the evaluation for adaptive CE

        For the rest of the attributes, please see the base class (LVISeval)
        """

        self.val_annotations = val_annotations
        self.test_annotations = test_annotations
        self.dataset_classes = list(LVIS(val_annotations).cats.keys())

        super(CalibrationLVIS, self).__init__(lvis_gt=LVIS(
            val_annotations), lvis_dt=None, iou_type=eval_type)

        # LVISEval related parameters
        self.params.area_rng = [self.params.area_rng[0]]
        self.params.area_rng_lbl = ['all']
        self.params.iou_thrs = np.array([tau])

        # usually max_dets=100 for COCO and max_dets=300 for LVIS
        self.params.max_dets = [max_dets]

        # evaluation-specific parameters
        self.tau = tau
        self.is_dece = is_dece
        self.is_ace = is_ace
        self.eval_type = eval_type
        self.bin_count = bin_count
        self.bins = np.linspace(0.0, 1.0, self.bin_count + 1, endpoint=True)

        # Calibrator-specific options, can be further set with fit()
        self.classagnostic = False
        self.calibrator_type = 'identity'

        self.calibration_info = dict()
        self.calibration_info_all = dict()

        # Follow D-ECE-style evaluation directly
        if self.is_dece:
            self.errors = np.zeros(self.bin_count)
            self.weights_per_bin = np.zeros(self.bin_count)
            self.prec_iou = np.zeros(self.bin_count)

        else:
            # For LaACE_0, follow class-wise and bin_width==1 strategy
            if self.is_ace:
                self.errors = np.zeros(len(self.params.cat_ids))
                self.weights_per_bin = np.zeros(len(self.params.cat_ids))
                self.prec_iou = np.zeros(len(self.params.cat_ids))

            # Else, follow LaECE-style binning
            else:
                self.errors = np.zeros(
                    [len(self.params.cat_ids), self.bin_count])
                self.weights_per_bin = np.zeros(
                    [len(self.params.cat_ids), self.bin_count])
                self.prec_iou = np.zeros(
                    [len(self.params.cat_ids), self.bin_count])

        self.lrps = {'lrp': np.zeros(len(self.params.cat_ids)) - 1, 'lrp_loc': np.zeros(len(self.params.cat_ids)) - 1,
                     'lrp_fp': np.zeros(len(self.params.cat_ids)) - 1, 'lrp_fn': np.zeros(len(self.params.cat_ids)) - 1}

    def prepare_input(self, p=None):
        """
        Accumulate per image evaluation results and
        store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        if not self.eval_imgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.cat_ids = p.cat_ids if p.use_cats == 1 else [-1]
        T = len(p.iou_thrs)
        R = len(p.rec_thrs)
        K = len(p.cat_ids) if p.use_cats else 1
        A = len(p.area_rng)
        M = len(p.max_dets)

        # create dictionary for future indexing
        _pe = copy.deepcopy(self.params)  # self._params_eval
        cat_ids = _pe.cat_ids if _pe.use_cats else [-1]
        setK = set(cat_ids)
        setA = set(map(tuple, _pe.area_rng))
        setM = set(_pe.max_dets)
        setI = set(_pe.img_ids)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.cat_ids) if k in setK]
        m_list = [m for n, m in enumerate(p.max_dets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.area_rng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.img_ids) if i in setI]
        I0 = len(_pe.img_ids)
        A0 = len(_pe.area_rng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            self.calibration_info[k] = dict()
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.eval_imgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dt_scores'][0:maxDet] for e in E])

                    # different sorting method generates slightly
                    # different results.
                    # mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dt_matches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                           inds]
                    dtIg = np.concatenate(
                        [e['dt_ignore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dtIoU = np.concatenate(
                        [e['dt_ious'][:, 0:maxDet] for e in E], axis=1)[:, inds]

                    gtIg = np.concatenate([e['gt_ignore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue

                    self.calibration_info[k]['scores'] = dtScoresSorted
                    self.calibration_info[k]['tps'] = np.logical_and(
                        dtm, np.logical_not(dtIg))[0]

                    self.calibration_info[k]['fps'] = np.logical_and(np.logical_not(dtm),
                                                                     np.logical_not(dtIg))[0]

                    self.calibration_info[k]['iou'] = np.multiply(
                        dtIoU, self.calibration_info[k]['tps'])[0]
                    self.calibration_info[k]['npig'] = npig

        return

    def combine_calibration_info(self):
        """
        Combines calibration info across classes
        for class-agnostic settings
        :return:
        """

        self.calibration_info_all['scores'] = np.array([])
        self.calibration_info_all['tps'] = np.array([])
        self.calibration_info_all['fps'] = np.array([])
        self.calibration_info_all['iou'] = np.array([])
        # Join all lists into one
        for cl, cl_input in self.calibration_info.items():

            # if no detections for that class, simply initialize the keys with []
            if len(cl_input.keys()) == 0:
                cl_input['scores'] = np.array([])
                cl_input['tps'] = np.array([])
                cl_input['fps'] = np.array([])
                cl_input['iou'] = np.array([])

            self.calibration_info_all['scores'] = np.append(
                self.calibration_info_all['scores'], cl_input['scores'])
            self.calibration_info_all['tps'] = np.append(
                self.calibration_info_all['tps'], cl_input['tps'])
            self.calibration_info_all['fps'] = np.append(
                self.calibration_info_all['fps'], cl_input['fps'])
            self.calibration_info_all['iou'] = np.append(
                self.calibration_info_all['iou'], cl_input['iou'])

        sorted_idx = (-self.calibration_info_all['scores']).argsort()
        self.calibration_info_all['scores'] = self.calibration_info_all['scores'][sorted_idx]
        self.calibration_info_all['tps'] = self.calibration_info_all['tps'][sorted_idx]
        self.calibration_info_all['fps'] = self.calibration_info_all['fps'][sorted_idx]
        self.calibration_info_all['iou'] = self.calibration_info_all['iou'][sorted_idx]
        return

    def compute_single_errors(self):
        """
        Computes single errors for each bins for
        either of the class-agnostic settings
        or the class-wise settings
        :return:
        """

        if self.is_dece:
            self.combine_calibration_info()
            # Find total number of valid detections for this class
            total_det = self.calibration_info_all['tps'].sum(
            ) + self.calibration_info_all['fps'].sum()

            # If no detection, then ignore
            if total_det == 0:
                return
            for i in range(self.bin_count):
                # Find detections in this bin

                if i == 0:
                    bin_all_det = np.logical_and(self.bins[i] <= self.calibration_info_all['scores'],
                                                 self.calibration_info_all['scores'] <= self.bins[i + 1])
                else:
                    bin_all_det = np.logical_and(self.bins[i] < self.calibration_info_all['scores'],
                                                 self.calibration_info_all['scores'] <= self.bins[i + 1])

                bin_tps = np.logical_and(
                    self.calibration_info_all['tps'], bin_all_det)
                bin_fps = np.logical_and(
                    self.calibration_info_all['fps'], bin_all_det)
                bin_det = np.logical_or(bin_tps, bin_fps)

                bin_scores = self.calibration_info_all['scores'][bin_det]

                # Count number of tps in this bin
                num_tp = bin_tps.sum()

                # Count number of fps in this bin
                num_fp = bin_fps.sum()

                # Count number of detections in this bin
                num_det = num_tp + num_fp

                if num_det == 0:
                    self.errors[i] = np.nan
                    self.weights_per_bin[i] = 0
                    self.prec_iou[i] = np.nan
                    continue
                else:
                    self.prec_iou[i] = num_tp / num_det

                # Average of Scores in this bin
                mean_score = bin_scores.mean()

                self.errors[i] = np.abs(self.prec_iou[i] - mean_score)

                # Weight of the bin
                self.weights_per_bin[i] = num_det / total_det

        # Class-wise evaluation
        else:
            for cl, cl_input in self.calibration_info.items():
                # For corrupted images, all images with this class is already rejected
                if 'tps' not in cl_input.keys():
                    continue

                # Find total number of valid detections for this class
                total_det = cl_input['tps'].sum() + cl_input['fps'].sum()

                # If no detection, then ignore
                if total_det == 0:
                    continue

                # LaACE_0 evaluation
                elif self.is_ace:

                    # Get the TP-FP information for all the detection of the current class
                    cl_tps = cl_input['tps'].sum()
                    cl_fps = cl_input['fps'].sum()
                    cl_dets = np.logical_or(cl_tps, cl_fps)

                    # Get the IoU and score information
                    ious = cl_input['iou'][cl_dets]
                    scores = cl_input['scores'][cl_dets]

                    # Accumulate the errors for the current class for eval
                    self.errors[cl] = np.mean(np.abs(ious - scores))

                # LaECE-style evaluation
                else:
                    for i in range(self.bin_count):
                        # Find detections in this bin

                        if i == 0:
                            bin_all_det = np.logical_and(self.bins[i] <= cl_input['scores'],
                                                         cl_input['scores'] <= self.bins[i + 1])
                        else:
                            bin_all_det = np.logical_and(self.bins[i] < cl_input['scores'],
                                                         cl_input['scores'] <= self.bins[i + 1])

                        bin_tps = np.logical_and(cl_input['tps'], bin_all_det)
                        bin_fps = np.logical_and(cl_input['fps'], bin_all_det)
                        bin_det = np.logical_or(bin_tps, bin_fps)
                        bin_scores = cl_input['scores'][bin_det]
                        bin_ious = cl_input['iou'][bin_tps]

                        # Count number of tps in this bin
                        num_tp = bin_tps.sum()

                        # Count number of fps in this bin
                        num_fp = bin_fps.sum()

                        # Count number of detections in this bin
                        num_det = num_tp + num_fp

                        if num_det == 0:
                            self.errors[cl, i] = np.nan
                            self.weights_per_bin[cl, i] = 0
                            self.prec_iou[cl, i] = np.nan
                            continue

                        # Find error
                        if len(bin_ious) > 0:
                            # norm_iou = (bin_ious - 0.10) / (1 - 0.10)
                            norm_iou = bin_ious
                            norm_total_iou = norm_iou.sum()
                        else:
                            norm_total_iou = 0

                        self.prec_iou[cl, i] = norm_total_iou / num_det

                        # Average of Scores in this bin
                        mean_score = bin_scores.mean()

                        self.errors[cl, i] = np.abs(
                            self.prec_iou[cl, i] - mean_score)

                        # Weight of the bin
                        self.weights_per_bin[cl, i] = num_det / total_det
        return

    def calibrate(self, detections, calibrator_type, calibration_model, lvis_classes):
        """
        Perform the calibration given detections and calibrator
        :param detections  (dict array)               : detections of the model
               calibrator_type  (str)                 : fitted calibrator type
               calibration_model (calibrator object)  : fitted calibrator object
               coco_classes (str array)               : dataset classes
        :return: detections (dict array)              : calibrated detections
        """

        if calibration_model is None:
            return detections

        if self.classagnostic:
            for detection in detections:
                if calibrator_type in ['linear_regression', 'isotonic_regression']:
                    if type(calibration_model) is not np.ndarray:
                        detection['score'] = \
                            np.clip(calibration_model.predict(
                                np.array(detection['score']).reshape(-1, 1)), 0, 1)[0]
                elif calibrator_type in ['platt_scaling', 'temperature_scaling']:
                    if type(calibration_model) is not np.ndarray:
                        transformed_conf = np.array(
                            [calibration_model.predict(np.array([detection['score']]))])
                        detection['score'] = np.clip(transformed_conf, 0, 1)[0]
        else:
            for detection in detections:
                cl = lvis_classes.index(detection['category_id'])
                if calibrator_type in ['linear_regression', 'isotonic_regression']:
                    if type(calibration_model[cl]) is not np.ndarray:
                        detection['score'] = \
                            np.clip(calibration_model[cl].predict(
                                np.array(detection['score']).reshape(-1, 1)), 0, 1)[0]
                elif calibrator_type in ['platt_scaling', 'temperature_scaling']:
                    if type(calibration_model[cl]) is not np.ndarray:
                        transformed_conf = np.array(
                            [calibration_model[cl].predict(np.array([detection['score']]))])
                        detection['score'] = np.clip(transformed_conf, 0, 1)[0]

        return detections

    def accumulate_errors(self):
        """
        Aggregate calibration errors across bins and/or classes
        :return: ECE (float)    : expected/adaptive calibration error
        """

        # Reports D-ECE
        if self.is_dece:
            ECE = np.nansum(self.weights_per_bin * self.errors)
        else:
            # Reports LaACE_0
            if self.is_ace:
                class_errors = self.errors
                class_errors[class_errors == 0] = np.nan
                ECE = np.nanmean(self.errors)
            # Reports LaECE_0 and LaECE depending on the tau
            else:
                bin_sum = np.nansum(self.weights_per_bin * self.errors, axis=1)
                bin_sum[bin_sum == 0] = np.nan
                ECE = np.nanmean(bin_sum)
        return ECE

    def plot_reliability_diagram(self, LaECE, cl=-1, fontsize=16):
        """
        Aggregate calibration errors across bins and/or classes
        :param ECE  (float)     : expected calibration error
               cl (int)         : class index
               fontsize (int)   : font size for the plot texts
        :return:
        """

        delta = 1.0 / self.bin_count
        x = np.arange(0, 1, delta)

        if cl == -1:
            bin_acc = np.nanmean(self.prec_iou, axis=0)
            bin_weights = np.nanmean(self.weights_per_bin, axis=0)
        else:
            bin_acc = self.prec_iou[cl]
            bin_weights = self.weights_per_bin[cl]
        nan_idx = (bin_weights == 0)
        bin_acc[nan_idx] = 0

        # size and axis limits
        plt.figure(figsize=(5, 5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plot grid
        plt.grid(color='tab:grey', linestyle=(
            0, (1, 5)), linewidth=1, zorder=0)
        # plot bars and identity line
        plt.bar(x, bin_acc, color='b', width=delta, align='edge', edgecolor='k',
                label=r'Precision $ \times \mathrm{\bar{IoU}}$',
                zorder=5)
        plt.bar(x, bin_weights, color='mistyrose', alpha=0.5, width=delta, align='edge',
                edgecolor='r', hatch='/', label='% of Samples', zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)
        # labels and legend
        plt.xlabel('Confidence', fontsize=fontsize)
        plt.legend(loc='upper left', framealpha=1.0, fontsize=fontsize)
        plt.text(0.05, 0.70, 'LaECE= %.1f%%' %
                 (LaECE * 100), fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()
        plt.show()
        return

    def compute_LRP(self):
        """
        Compute LRP based on the class-wise TP/FP/localization information
        :return:
        """

        for cl, cl_input in self.calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                self.lrps['lrp_loc'][cl] = np.nan
                self.lrps['lrp_fp'][cl] = np.nan
                self.lrps['lrp_fn'][cl] = np.nan
                self.lrps['lrp'][cl] = np.nan
                continue

            # Find total number of valid detections for this class
            tp_num = cl_input['tps'].sum()
            fp_num = cl_input['fps'].sum()
            fn_num = cl_input['npig'] - tp_num

            # If there is detection
            if tp_num + fp_num > 0:
                # There is some TPs
                if tp_num > 0:
                    total_loc = tp_num - cl_input['iou'].sum()
                    self.lrps['lrp'][cl] = (total_loc / (1 - self.tau) + fp_num +
                                            fn_num) / (tp_num + fp_num + fn_num)
                    self.lrps['lrp_loc'][cl] = total_loc / tp_num
                    self.lrps['lrp_fp'][cl] = fp_num / (tp_num + fp_num)
                    self.lrps['lrp_fn'][cl] = fn_num / cl_input['npig']
                else:
                    self.lrps['lrp_loc'][cl] = np.nan
                    self.lrps['lrp_fp'][cl] = np.nan
                    self.lrps['lrp_fn'][cl] = 1.
                    self.lrps['lrp'][cl] = 1.
            else:
                self.lrps['lrp_loc'][cl] = np.nan
                self.lrps['lrp_fp'][cl] = np.nan
                self.lrps['lrp_fn'][cl] = 1.
                self.lrps['lrp'][cl] = 1.
        return

    def fit(self, val_detections, calibrator_type, thresholds=[-1, -1], classagnostic=False, use_shift=True, use_grid_search=False, use_quality_focal_loss=False):
        """
        Fit a specified calibrator based on specified thresholds
        :param val_detections  (dict array)                 : validation detections
               calibrator_type  (str)                        : fitted calibrator type
               thresholds (float array)                      : pair of thresholds for two stages
               eval_type (str)                               : evaluation type, either 'bbox' or 'segm
               classagnostic (bool)                          : class-agnostic or class-wise calibration
               use_grid_search (bool)                        : grid searching or L-BFGS optimizing TS/PS
               use_quality_focal_loss (bool)                 : objective for optimizing TS/PS
        :return: calibration_model, [pre_thr, operating_thr] : fitted calibrator, [first thresholds, second thresholds]
        """

        # set the calibrator-specific parameters
        self.calibrator_type = calibrator_type
        self.classagnostic = classagnostic

        val_detections = load_detections_from_file(val_detections)
        dataset_classes = self.dataset_classes

        # the following corresponds to the first set of thresholds learned pre-calibration stage
        pre_calibration_thresholds = get_detection_thresholds(
            self.val_annotations, val_detections, 'lvis', thresholds[0], self.tau, self.eval_type, max_dets=self.params.max_dets)

        thresholded_val_detections = threshold_detections(
            val_detections, pre_calibration_thresholds, dataset_classes)
        self.lvis_dt = LVISResults(
            self.val_annotations, thresholded_val_detections)
        self.iou_type = self.eval_type

        self.evaluate()
        self.prepare_input()

        if self.classagnostic:
            self.combine_calibration_info()
            calibration_info = self.calibration_info_all
        else:
            calibration_info = self.calibration_info

        if calibrator_type == 'identity':
            calibration_model = None
        elif calibrator_type == 'linear_regression':
            calibration_model = linear_regression(
                self.classagnostic, calibration_info, self.is_dece)
        elif calibrator_type == 'isotonic_regression':
            calibration_model = isotonic_regression(
                self.classagnostic, calibration_info, self.is_dece)
        elif calibrator_type == 'platt_scaling':
            calibration_model = platt_scaling(self.classagnostic, calibration_info, self.is_dece,
                                              use_grid_search, use_quality_focal_loss)
        elif calibrator_type == 'temperature_scaling':
            calibration_model = temperature_scaling(self.classagnostic, calibration_info, self.is_dece,
                                                    use_grid_search, use_quality_focal_loss)
        else:
            raise Exception(
                f"Calibration method {calibrator_type} not implemented. Please choose one from ['identity', 'linear_regression', 'isotonic_regression', 'platt_scaling', 'temperature_scaling']")

        calibrated_val_detections = self.calibrate(
            val_detections, calibrator_type, calibration_model, dataset_classes)

        # the following corresponds to the second set of thresholds that should ideally be used for the operating stage
        operating_thresholds = get_detection_thresholds(
            self.val_annotations, calibrated_val_detections, 'lvis', thresholds[1], self.tau, self.eval_type, max_dets=self.params.max_dets)

        return calibration_model, thresholds

    def transform(self, test_detections, calibration_model, thresholds, show_plot=False):
        """
        Calibrate a given set of detections based on a fitted calibrator
        :param test_detections  (dict array)                : test detections
               calibration_model  (calibrator object)       : fitted calibrator type
               thresholds (float array)                     : pair of thresholds for two stages
        :return: calibrated_test_detections                 : calibrated test detections
        """

        test_detections = load_detections_from_file(test_detections)

        # May or may not threshold test detections before the calibration step
        thresholded_test_detections = threshold_detections(
            test_detections, thresholds[0], self.dataset_classes)

        # Calibrate test data detections, whether all or only survived
        calibrated_test_detections = self.calibrate(
            thresholded_test_detections, self.calibrator_type, calibration_model, self.dataset_classes)

        # Re-threshold the calibrated detections for a fair comparison
        calibrated_test_detections = threshold_detections(
            calibrated_test_detections, thresholds[1], self.dataset_classes)

        return calibrated_test_detections

    def evaluate_calibration(self, calibrated_test_detections, is_dece=False, show_plot=False, verbose=False):
        """
        Joint evaluation of accuracy and calibration
        :param calibrated_test_detections  (dict array)     : calibrated test detections
               is_dece  (bool)                              : print D-ECE or not
               show_plot (bool)                             : save the reliability diagram or not
               verbose (bool)                               : print AP components or not
        :return: calibrated_test_detections                 : calibrated test detections
        """

        # Compute the joint evaluation measures of performance and calibration
        calibration_eval = CalibrationLVIS(
            self.test_annotations, self.test_annotations, self.eval_type, 25, 0.0, False, False, self.params.max_dets[0])

        calibration_eval.lvis_dt = LVISResults(
            self.test_annotations, calibrated_test_detections)
        calibration_eval.evaluate()
        calibration_eval.prepare_input()
        calibration_eval.compute_single_errors()
        LaECE_0 = calibration_eval.accumulate_errors()

        # The final True is for the LaACE option, as it does not require an explicit set of binning
        calibration_eval = CalibrationLVIS(self.test_annotations, self.test_annotations, self.eval_type, self.bin_count, 0.0,
                                           False, True, self.params.max_dets[0])

        calibration_eval.lvis_dt = LVISResults(
            self.test_annotations, calibrated_test_detections)
        calibration_eval.evaluate()
        calibration_eval.prepare_input()
        calibration_eval.compute_single_errors()
        calibration_eval.compute_LRP()
        LaACE_0 = calibration_eval.accumulate_errors()

        print()
        print('--------------------------ACCURACY-------------------------')
        print(
            f'LRP       @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {np.nanmean(calibration_eval.lrps["lrp"]) * 100:.1f}')
        print(
            f'LRP Loc   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {np.nanmean(calibration_eval.lrps["lrp_loc"]) * 100:.1f}')
        print(
            f'LRP FP    @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {np.nanmean(calibration_eval.lrps["lrp_fp"]) * 100:.1f}')
        print(
            f'LRP FN    @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {np.nanmean(calibration_eval.lrps["lrp_fn"]) * 100:.1f}')

        print('\n-------------------------CALIBRATION-----------------------')

        if self.tau == 0.0:
            print(
                f'LaECE_0   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {LaECE_0 * 100:.1f}')
            print(
                f'LaACE_0   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {LaACE_0 * 100:.1f}')
        else:
            print(
                f'LaECE     @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {LaECE_0 * 100:.1f}')
            print(
                f'LaACE     @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {LaACE_0 * 100:.1f}')

        if is_dece:
            calibration_eval = CalibrationLVIS(
                self.test_annotations, self.test_annotations, self.eval_type, 10, 0.5, True, False, self.params.max_dets[0])

            calibration_eval.lvis_dt = LVISResults(
                self.test_annotations, calibrated_test_detections)
            calibration_eval.evaluate()
            calibration_eval.prepare_input()
            calibration_eval.compute_single_errors()
            DECE = calibration_eval.accumulate_errors()
            print(
                f'D-ECE     @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.max_dets} ] = {DECE * 100:.1f}\n')
        else:
            print('\n')

        if verbose:
            LVIS_evaluation(self.test_annotations,
                            calibrated_test_detections, self.eval_type)

        # show_plot reliability diagrams only supporst LaECE_0
        if show_plot:
            self.plot_reliability_diagram(LaECE_0, cl=-1, fontsize=22)

        return
