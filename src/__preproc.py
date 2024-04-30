import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns 
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
    def __init__(self):
        self.df: pd.core.frame.DataFrame = dataset
        self.original_features: list[str] = self.df.columns
        self.corr: pd.core.series.Series = correlation
    
    def __feature_select(self,drop_corr:bool=False) -> None:
        self.feature_list:list[str] = []
        if drop_corr:
            for feature in self.original_features:
                for value in self.corr[feature]:
                    if (value > 0.92) & (feature not in self.feature_list):
                        self.feature_list.append(feature)
            print(self.feature_list)
        self.feature_list:list[str] = list(self.corr[abs(self.corr['diagnosis']) > 0.59].index)
        print(self.feature_list)
    
    def __feature_select_save(self) -> None:
        eliminate_features = [feature for feature in self.original_features if feature not in self.feature_list]
        self.df.drop(eliminate_features,axis=1,inplace=True)
    
    def main(self)->list[str]:
        self.__feature_select()
        #self.__feature_select(drop_corr=True)
        self.__feature_select_save()
        return self.df


class RemoveOutlier:
    pass

class Standardizer:
    def __init__(self) ->None:
        self.scaler = StandardScaler()
        self.X = dataset.iloc[:,2:]
    def __scale(self) ->None:
        self.X = self.scaler.fit_transform(self.X)
        return self.X
    def main(self):
        return self.__scale()
class DimensionReduce:
    def __init__(self) -> None:
        self.df = dataset
        self.feature_list:list[str] = list(self.df.columns)
        self.feature_list.remove('diagnosis')
        self.label = 'diagnosis'
        self.scaler = Standardizer()
        self.pca = PCA(n_components=5)
    
    def __reduce_dim(self):
        self.X = self.scaler.main()
        self.trans = self.pca.fit_transform(self.X)
        print(self.pca.get_feature_names_out())
        self.output_df = pd.DataFrame(self.trans,columns=['PC1', 'PC2', 'PC3','PC4','PC5'])
        self.output_df['diagnosis'] = self.df[self.label]
        self.pca_features = self.output_df.columns
        
    def __plot(self):
        fig,axes= plt.subplots(5,5,figsize=(25,20),squeeze=False)
        fig.suptitle('PCA plot')
        for row in range(5):
            featurex = self.pca_features[row]
            x = self.output_df[featurex]
            for col in range(5):
                featurey = self.pca_features[col]
                y = self.output_df[featurey]
                sns.scatterplot(data=self.output_df, x=x,y=y,ax=axes[row][col],hue=self.label,palette=sns.color_palette('viridis')[:2])
        plt.tight_layout()
        plt.savefig(r'plots\pca.png')
    
    def __return_transformed(self):
        return self.output_df
    def main(self):
        self.__reduce_dim()
        #self.__plot()
        return self.__return_transformed()
 

if __name__ == '__main__':
    fs = FeatureSelect()
    fs.main()
    dr = DimensionReduce()
    dr.main()
    