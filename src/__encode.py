import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv(r'datasets\breast-cancer.csv')
class Encoder:
    def __init__(self, dataframe:pd.core.frame.DataFrame)->None:
        self.df = dataframe
        self.label = 'diagnosis'
    
    def __encode_label(self) -> pd.core.series.Series:
        self.encoder = LabelEncoder()
        return pd.Series(self.encoder.fit_transform(self.df[self.label]))
    
    def encode(self):
        return self.__encode_label()
        
    
if __name__ == '__main__':
    enc = Encoder(dataset)
    enc.encode()