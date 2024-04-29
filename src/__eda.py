import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from __import import Import
from __numerical import ReturnAnalysis
from __encode import Encoder

import_object = Import()
encoder_obj = Encoder(import_object.main())


bv_object = ReturnAnalysis()


palette = sns.cubehelix_palette(start=-0.15,rot=-0.75)
cmap = sns.cubehelix_palette(start=-0.15,rot=-0.75,as_cmap=True)
sns.set_palette(palette=palette)
class EDA:
    def __init__(self) -> None:
        self.df = import_object.main()
        self.feature_list:list[str] = list(self.df.columns)
        self.temp_list = self.feature_list[1:]
        self.label = 'diagnosis'
        self.df[self.label] = encoder_obj.encode()
    
    def __boxplot(self) -> None:
        fig,axes = plt.subplots(3,10,figsize=(40,10), squeeze=False)
        fig.suptitle('BoxPlots for outlier Detection')
        count = 0
        for row in range(3):
            for col in range(10):
                feature = self.temp_list[count]
                y = self.df[feature]
                sns.boxplot(data=self.df, y=y,width=0.2,ax=axes[row][col],hue=self.label,palette=palette)
                axes[row][col].set_title(f'Boxplot for {feature}')
                axes[row][col].set_xlabel(f'{feature}')
                count += 1
        plt.tight_layout()
        plt.savefig(r'plots\boxplot.png')
        
    def __normal_distr(self) ->None:
        fig,axes = plt.subplots(3,10,figsize=(40,10),squeeze=False)
        fig.suptitle('Normal Distribution Curve')
        count:int = 0
        for row in range(3):
            for col in range(10):
                feature = self.temp_list[count]
                x:pd.core.series.Series = self.df[feature]
                sns.histplot(data=self.df, x=x, kde=True,ax=axes[row][col]).lines[0].set_color('black')
                axes[row][col].set_title(f'N Distr {feature}')
                axes[row][col].set_xlabel(f'{feature}')
                axes[row][col].set_ylabel('Frequency')
                count += 1
        plt.tight_layout()
        plt.savefig(r'plots\normaldistr.png')
    
    def __scatterplot(self) -> None:
        """
            cherry picked attributes with either really high covariance, really low covariance or just average
        """
        
        
    
    def __heatplot(self)  -> None:
        fig,axes = plt.subplots(1,1,figsize=(30,20),squeeze=False)
        fig.suptitle('Heatplot for variables')
        #self.corr = bv_object.main()
        self.corr = self.df.corr()
        sns.heatmap(data=self.corr, ax=axes[0][0], center=0, annot=True,cmap=cmap)
        plt.tight_layout()
        plt.savefig(r'plots\heatmap.png')
        
        
    
    def __clustermap(self) -> None:
        fig, axes = plt.subplots(1,1,figsize=(20,20),squeeze=False)
        fig.suptitle('Cluster Map')
        sns.clustermap(data=self.corr,  center=0, cmap=cmap)
        plt.tight_layout()
        plt.savefig(r'plots\clustermap.png')
    
        
        
                        
        
        
    
    def main(self) -> None:
        self.__boxplot()
        self.__normal_distr()
        self.__heatplot()
        self.__clustermap()
        
            
                
        
    
    
if __name__ == '__main__':
    eda = EDA()
    eda.main()