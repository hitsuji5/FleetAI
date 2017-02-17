import numpy as np

class Demand(object):
    def __init__(self, model):
        self.model = model

    def predict(self, df):
        return self.model.predict(df.values)
