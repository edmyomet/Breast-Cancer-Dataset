import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from __encode import Encoder
from __import import Import
from __numerical import ReturnAnalysis

import_obj = Import()
dataset = import_obj.main()

encode_obj = Encoder(dataset)
dataset['diagnosis'] = encode_obj.encode()

numeric_analysis_obj = ReturnAnalysis()
correlation = numeric_analysis_obj.main()

class FeatureSelect:
    pass


class RemoveOutlier:
    pass

class Normaliser:
    pass