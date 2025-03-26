import numpy as np
from scipy.special import logit as safe_logit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def quality_focal_loss(logits, targets, beta=1):
    confidences = torch.sigmoid(logits)
    scale_factor = torch.pow(torch.abs(confidences - targets), exponent=beta)

    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')*scale_factor

    return loss.mean(dim=0)


class PlattScaling(nn.Module):
    def __init__(self):
        """
        Class for learning a PlattScaling calibrator

        Attributes:
            device (str)       : device to perform fit() and transform()
            scale (Parameter)  : scale parameter in PS, corresponds to a in a*z + b
            shift (Parameter)  : shift parameter in PS, corresponds to b in a*z + b
        """

        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def inverse_sigmoid(self, confidences):
        """
        Perform inverse sigmoid on the confidences to obtain the logits
        :param confidences  (float array)   : confidence scores of the detections
        :return: logits (float array)       : logits of the detections (inv. sigmoid of p_i vals)
        """

        # on torch tensors, use torch built-in functions
        if isinstance(confidences, torch.Tensor):
            epsilon = float(torch.finfo(torch.float64).eps)

            # clip normal and inverse separately due to numerical stability
            clipped = torch.clamp(confidences, epsilon, 1. - epsilon)
            inv_clipped = torch.clamp(1. - confidences, epsilon, 1. - epsilon)

            logits = torch.log(clipped) - torch.log(inv_clipped)
            return logits

        # use NumPy method otherwise
        else:
            epsilon = float(np.finfo('float64').eps)
            clipped = np.clip(confidences, epsilon, 1. - epsilon)
            return safe_logit(clipped)

    def cross_validate_params(self, logits, targets, use_quality_focal_loss):
        """
        Cross validate the scale and shift params to obtain the best fit
        :param logits  (float array)           : inv. sigmoid of the confidence scores of the detections
               targets (float array)           : calibration targets for fitting the calibrator
               use_quality_focal_loss (bool)   : objective for optimizing TS/PS 
        :return: self
        """

        lower_bound = -5
        upper_bound = 5
        num_values = 100

        # Range and step size for grid searching the scale
        lower = lower_bound + upper_bound
        upper = upper_bound * 2
        scale_range = np.linspace(lower, upper, num=num_values)

        # Range and step size for grid searching the shift
        shift_lower = lower_bound
        shift_upper = upper_bound
        shift_range = np.linspace(shift_lower, shift_upper, num=num_values)

        min_error = 1e32

        for scale in scale_range:
            for shift in shift_range:
                _logits = self.transform(
                    logits, scale_val=scale, shift_val=shift)

                if use_quality_focal_loss:
                    loss = quality_focal_loss(_logits, targets)

                else:
                    loss = F.binary_cross_entropy_with_logits(
                        _logits, targets, reduction='mean')

                if loss < min_error:
                    min_error = loss
                    with torch.no_grad():
                        self.scale = nn.Parameter(
                            torch.tensor(scale), requires_grad=False)
                        self.shift = nn.Parameter(
                            torch.tensor(shift), requires_grad=False)

        return self

    def optimize_params(self, logits, targets, use_quality_focal_loss):
        """
        Use L-BFGS optimization for the scale and shift params to obtain the best fit
        :param logits  (float array)           : inv. sigmoid of the confidence scores of the detections
               targets (float array)           : calibration targets for fitting the calibrator
               use_quality_focal_loss (bool)   : objective for optimizing TS/PS 
        :return: self
        """

        optimizer = optim.LBFGS(
            [self.scale, self.shift], lr=1e-1, max_iter=100)

        def eval():
            optimizer.zero_grad()

            if use_quality_focal_loss:
                loss = quality_focal_loss(self.transform(logits), targets)

            else:
                loss = F.binary_cross_entropy_with_logits(
                    self.transform(logits), targets, reduction='mean')

            loss.backward()
            return loss
        optimizer.step(eval)
        return self

    def fit(self, confidences, targets, use_grid_search=False, use_quality_focal_loss=False):
        """
        Fit the PS post-hoc calibrator
        :param confidences  (float array)      : confidence scores of the detections
               targets (float array)           : calibration targets for fitting the calibrator
               use_grid_search (bool)          : grid searching or L-BFGS optimizing TS/PS
               use_quality_focal_loss (bool)   : objective for optimizing TS/PS 
        :return: self
        """

        logits = self.inverse_sigmoid(confidences)
        logits = torch.tensor(logits, requires_grad=False).squeeze()

        if logits.size() == []:
            print(f'Had no predictions, halting...')
            return -1
        targets = torch.tensor(targets, requires_grad=False)

        # Continue with grid searching for the parameters
        if use_grid_search:
            return self.cross_validate_params(logits, targets, use_quality_focal_loss)

        # Continue with L-BFGS optimization
        else:
            return self.optimize_params(logits, targets, use_quality_focal_loss)

    def transform(self, logits, scale_val=None, shift_val=None):
        """
        Calibrate the test detections based on the logits with the PS post-hoc calibrator
        :param logits  (float array)  : inv. sigmoid of the confidence scores of the detections
               scale_val (float)      : scale parameter in PS, corresponds to a in a*z + b
               shift_val (float)      : shift parameter in PS, corresponds to b in a*z + b
        :return: self
        """

        if logits.size() == torch.Size([]):
            logits = logits.unsqueeze(dim=0)

        if scale_val is not None:
            scale = torch.tensor(scale_val)
        else:
            scale = self.scale.expand(logits.size())

        if shift_val is not None:
            shift = torch.tensor(shift_val)
        else:
            shift = self.shift.expand(logits.size())

        return logits * torch.abs(scale) + shift

    def predict(self, confidences):
        """
        Perform prediction on the confidences with the PS post-hoc calibrator
        :param confidences  (float array)  : confidence scores of the detections
        :return: float                     : calibrated confidence score (single value)
        """

        logits = torch.tensor(self.inverse_sigmoid(confidences))

        if logits.size() == torch.Size([]):
            logits = logits.unsqueeze(dim=0)

        scale = self.scale.expand(logits.size())
        logits *= torch.abs(scale)

        shift = self.shift.expand(logits.size())
        logits += shift

        # fetch scalar value.
        return torch.sigmoid(logits).detach().item()


class TemperatureScaling(nn.Module):
    def __init__(self):
        """
        Class for learning a TemperatureScaling calibrator

        Attributes:
            device (str)             : device to perform fit() and transform()
            temperature (Parameter)  : temperature parameter, corresponds to T in p_i/T
        """

        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)

    def inverse_sigmoid(self, confidences):
        """
        Perform inverse sigmoid on the confidences to obtain the logits
        :param confidences  (float array)   : confidence scores of the detections
        :return: logits (float array)       : logits of the detections (inv. sigmoid of p_i vals)
        """

        # on torch tensors, use torch built-in functions
        if isinstance(confidences, torch.Tensor):
            epsilon = float(torch.finfo(torch.float64).eps)

            # clip normal and inverse separately due to numerical stability
            clipped = torch.clamp(confidences, epsilon, 1. - epsilon)
            inv_clipped = torch.clamp(1. - confidences, epsilon, 1. - epsilon)

            logit = torch.log(clipped) - torch.log(inv_clipped)
            return logit

        # use NumPy method otherwise
        else:
            epsilon = float(np.finfo('float64').eps)
            clipped = np.clip(confidences, epsilon, 1. - epsilon)
            return safe_logit(clipped)

    def cross_validate_params(self, logits, targets, use_quality_focal_loss):
        """
        Cross validate the temperature parameter to obtain the best fit
        :param logits  (float array)           : inv. sigmoid of the confidence scores of the detections
               targets (float array)           : calibration targets for fitting the calibrator
               use_quality_focal_loss (bool)   : objective for optimizing TS/PS 
        :return: self
        """

        lower = self.lower_bound
        upper = self.upper_bound
        num_values = self.num_values
        temp_range = np.linspace(lower, upper, num=num_values)

        min_error = 1e32

        for temp in temp_range:
            _logits = self.transform(logits, temp_val=temp)

            if use_quality_focal_loss:
                loss = quality_focal_loss(_logits, targets)

            else:
                loss = F.binary_cross_entropy_with_logits(
                    _logits, targets, reduction='mean')

            if loss < min_error:
                min_error = loss
                with torch.no_grad():
                    self.temp = nn.Parameter(
                        torch.tensor(temp), requires_grad=False)

        return self

    def optimize_params(self, logits, targets, use_quality_focal_loss):
        """
        Use L-BFGS optimization for the temperature parameter to obtain the best fit
        :param logits  (float array)           : inv. sigmoid of the confidence scores of the detections
               targets (float array)           : calibration targets for fitting the calibrator
               use_quality_focal_loss (bool)   : objective for optimizing TS/PS 
        :return: self
        """

        optimizer = optim.LBFGS([self.temperature], lr=1e-1, max_iter=100)

        def eval():
            optimizer.zero_grad()

            if use_quality_focal_loss:
                loss = quality_focal_loss(self.transform(logits), targets)

            else:
                loss = F.binary_cross_entropy_with_logits(
                    self.transform(logits), targets, reduction='mean')

            loss.backward()
            return loss

        optimizer.step(eval)
        return self

    def fit(self, confidences, targets, use_grid_search=False, use_quality_focal_loss=False):
        """
        Fit the TS post-hoc calibrator
        :param confidences  (float array)      : confidence scores of the detections
               targets (float array)           : calibration targets for fitting the calibrator
               use_grid_search (bool)          : grid searching or L-BFGS optimizing TS/PS
               use_quality_focal_loss (bool)   : objective for optimizing TS/PS 
        :return: self
        """

        logits = self.inverse_sigmoid(confidences)
        logits = torch.tensor(logits, requires_grad=False).squeeze()

        if logits.size() == []:
            print(f'Had no predictions, halting...')
            return -1
        targets = torch.tensor(targets, requires_grad=False)

        # Continue with grid searching for the parameters
        if use_grid_search:
            return self.cross_validate_params(logits, targets, use_quality_focal_loss)

        # Continue with L-BFGS optimization
        else:
            return self.optimize_params(logits, targets, use_quality_focal_loss)

    def transform(self, logits, temp_val=None):
        """
        Calibrate the test detections based on the logits with the TS post-hoc calibrator
        :param logits  (float array)    : inv. sigmoid of the confidence scores of the detections
               temperature (Parameter)  : temperature parameter, corresponds to T in p_i/T
        :return: self
        """

        if logits.size() == torch.Size([]):
            logits = logits.unsqueeze(dim=0)

        if temp_val is not None:
            temperature = torch.tensor(temp_val)
        else:
            temperature = self.temperature.expand(logits.size())

        return logits / torch.abs(temperature)

    def predict(self, confidences):
        """
        Perform prediction on the confidences with the TS post-hoc calibrator
        :param confidences  (float array)  : confidence scores of the detections
        :return: float                     : calibrated confidence score (single value)
        """

        logits = torch.tensor(self.inverse_sigmoid(confidences))

        if logits.size() == torch.Size([]):
            logits = logits.unsqueeze(dim=0)

        temp = self.temperature.expand(logits.size())
        logits /= torch.abs(temp)

        # fetch scalar value.
        return torch.sigmoid(logits).detach().item()

# Thanks to https://github.com/gpleiss/temperature_scaling/
