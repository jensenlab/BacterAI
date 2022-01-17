from abc import ABC, abstractmethod
from enum import Enum
import os

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as rpyn
from rpy2.robjects.packages import STAP
import torch

from global_vars import *
import net


class ModelType(Enum):
    GPR = 0
    NEURAL_NET = 1


class Model(ABC):
    def __init__(self, model, model_type):
        self.model_type = model_type

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def close(self):
        pass

    def get_type(self):
        return self.model_type

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X):
        pass


class GPRModel(Model):
    def __init__(self):
        self.activate_R()
        self.is_trained = False
        super().__init__(self, ModelType.GPR)

    def train(self, X_train, y_train):
        X_trainR = robjects.r.matrix(
            X_train, nrow=X_train.shape[0], ncol=X_train.shape[1]
        )
        y_trainR = robjects.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)
        self.model = self.gpr_lib.train_new_GP(X_trainR, y_trainR)
        self.is_trained = True

    def evaluate(self, X, clip=True, n=1):
        X_evalR = robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        if not self.is_trained:
            raise Exception("GPR model needs to be trained before evaluating.")

        resultR = self.gpr_lib.sample_GP(self.model, X_evalR, n)
        result = np.array(resultR)
        if clip:
            result = np.clip(result, 0, 1)
        samples, variances = result[:, 0], result[:, 1]
        return samples, variances

    def activate_R(self):
        with open("gpr_lib.R", "r") as f:
            s = f.read()
            self.gpr_lib = STAP(s, "gpr_lib")
            robjects.r("Sys.setenv(MKL_DEBUG_CPU_TYPE = '5')")
        rpyn.activate()

    def close(self):
        # Clean up R's GPR model object
        self.gpr_lib.delete_GP(self.model)


class NeuralNetModel(Model):
    def __init__(self, models_path):
        self.models_path = models_path
        self.models = []
        self.is_trained = False
        super().__init__(self, ModelType.NEURAL_NET)

    @classmethod
    def load_trained_models(cls, models_path):
        obj = cls(models_path)

        for filename in os.listdir(models_path):
            if "bag_model" in filename:
                model = torch.load(os.path.join(models_path, filename))
                obj.models.append(model)

        obj.is_trained = True
        return obj

    def check_path(self):
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train(self, X_train, y_train, **kwargs):
        self.check_path()
        self.models = net.train_bagged(X_train, y_train, self.models_path, **kwargs)
        self.is_trained = True

    def evaluate(self, X, clip=True):
        if not self.is_trained:
            raise Exception("Neural net model needs to be trained before evaluating.")

        predictions, variances = net.eval_bagged(X, self.models)
        if clip:
            predictions = np.clip(predictions, 0, 1)
        return predictions, variances
