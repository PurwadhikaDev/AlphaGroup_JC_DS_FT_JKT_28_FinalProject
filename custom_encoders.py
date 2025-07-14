from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import BinaryEncoder
import pandas as pd

class BinaryEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to make BinaryEncoder fully sklearn-compatible.
    Ensures compatibility with pipelines, pickling, and feature name propagation.
    """
    def __init__(self):
        self.encoder = BinaryEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def fit_transform(self, X, y=None):
        return self.encoder.fit_transform(X, y)

    def get_feature_names_out(self, input_features=None):
        try:
            return self.encoder.get_feature_names_out(input_features)
        except AttributeError:
            # Fallback for older versions of category_encoders
            return self.encoder.get_feature_names()
