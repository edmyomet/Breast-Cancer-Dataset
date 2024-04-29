import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from __encode import Encoder



class UnivariateAnalysis:
    def __init__(self) -> None:
        self.df:pd.core.frame.DataFrame = pd.read_csv(r'datasets\breast-cancer.csv')
        self.feature_list:list[str] = list(self.df.columns)
        self.feature_list.remove('diagnosis')
        self.encoder = Encoder(self.df)
        self.label:str = self.df.columns[1]
    
    def __info(self) ->None:
        print(self.df.info())
    
    def __descr(self,feature:str='descr') -> None:
        if feature != 'descr':
            self.df.head().to_csv(r'output\head.csv')
            self.df.tail().to_csv(r'output\tail.csv')
        self.df.describe().to_csv(r'output\description.csv')
    
    def __freq(self) -> None:
        for feature in self.feature_list:
            self.df[feature].value_counts().to_csv(r'output\frequency.csv',mode='a')
            """
                AS every value has occured once, computing the frequency is not a very useful measure. 
                anomalies if present have to be removed using the IQR method, and null values must be removed by replacing with mean or median
            """
    def __mean(self)->None:
        self.mean:pd.core.series.Series = self.df.mean(numeric_only=True)
        self.mean.to_csv(r'output\mean.csv')
    
    def __median(self)->None:
        self.median:pd.core.series.Series = self.df.median(numeric_only=True)
        self.median.to_csv(r'output\median.csv')
    
    def __mode(self)->None:
        self.mode:pd.core.series.Series = self.df.mode(numeric_only=True)
        self.mode.to_csv(r'output\mode.csv')
    
    def __std(self)->None:
        self.std:pd.core.series.Series = self.df.std(numeric_only=True)
        self.std.to_csv(r'output\std.csv')
    
    def __skew(self)->None:
        self.skew:pd.core.series.Series = self.df.skew(numeric_only=True)
        self.skew.to_csv(r'output\skew.csv')
    
    def __kurt(self)->None:
        self.kurt:pd.core.series.Series = self.df.kurt(numeric_only=True)
        self.kurt.to_csv(r'output\kurt.csv')

    def main(self)->None:
        #self.__info()
        self.__descr()
        self.__descr(feature='notdescr')
        self.__freq()
        self.__mean()
        self.__median()
        self.__mode()
        self.__skew()
        self.__kurt()
        self.__std()
            



class BivariateAnalysis(UnivariateAnalysis):
    def __init__(self) -> None:
        super().__init__()
        
    def __cov(self) -> None:
        self.cov : pd.core.series.Series = self.df.cov(numeric_only=True)
        self.cov.to_csv(r'output\covariance_matrix.csv')
    
    def __corr(self) -> pd.core.series.Series:
        self.corr:pd.core.series.Series = self.df.corr(numeric_only=True)
        self.corr.to_csv(r'output\correlation.csv')
        return self.corr

    
    def __regr(self):
        y:pd.core.series.Series = self.encoder.encode()
        self.regr = LinearRegression()
        self.output:dict[dict[str]] = {output_feature : {'Coeff': None, 'Intercept':None} for output_feature in self.feature_list}
        for feature in self.feature_list:
            X:pd.core.frame.DataFrame = pd.DataFrame(self.df[feature])
            self.regr.fit(X, y)
            self.output[feature]['Coeff'] = self.regr.coef_
            self.output[feature]['intercept'] = self.regr.intercept_
        
        pd.DataFrame.from_dict(self.output,orient='index').to_csv(r'output\regr.csv')
    
    def main(self):
        self.__cov()
        self.__corr()
        self.__regr()
            

class ReturnAnalysis(UnivariateAnalysis):
    def __init__(self) -> None:
        super().__init__()
        self.df['diagnosis'] = self.encoder.encode()

    def __return_corr(self) -> pd.core.series.Series:
        return self.df.corr()

    def main(self) -> None:
        return self.__return_corr()
            
if __name__ == '__main__':
    uv = UnivariateAnalysis()
    uv.main()
    bv = BivariateAnalysis()
    bv.main()
    
    ra = ReturnAnalysis()
    print(ra.main())