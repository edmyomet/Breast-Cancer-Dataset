import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score, accuracy_score
from __preproc import DimensionReduce, FeatureSelect
from __encode import Encoder
pca = DimensionReduce()
pca_dataset = pca.main()

feature_select = FeatureSelect()
dataset = feature_select.main()

encoder = Encoder(dataframe=dataset)
encoder.encode()

class DecisionTree:
    def __init__(self,reduced:bool=False):
        if reduced:
            self.df = pca_dataset
            self.feature_list = list(self.df.columns)
            self.feature_list.remove('diagnosis')
            self.X = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1:]
        else:
            self.df = dataset
            self.feature_list = list(self.df.columns)
            self.y = self.df.loc[:,'diagnosis']
            self.df.drop('diagnosis',inplace=True,axis=1)
            self.X = self.df.iloc[:,:]
    
    def __train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=7)
    
    def __build_model(self):
        fig = plt.figure(figsize=(30,8))
        self.classifier = DecisionTreeClassifier(splitter='random',random_state=2021)
        self.classifier.fit(self.X_train,self.y_train)
        plot_tree(self.classifier,filled=True)
        plt.savefig(r'plots\decisiontree.png')
        
    def __obtain_metadata(self):
        self.classifier.get_n_leaves()
        self.classifier.get_depth()
        self.classifier.decision_path(self.X_train)
    
    def __predict(self,user:bool=False,X:np.ndarray=None):
        if user:
            
            return self.classifier.predict(X)
        return self.classifier.predict(self.X_test)
    

    def __evaluate(self):
        self.y_pred = self.__predict()
        #print(encoder.encoder.inverse_transform(self.y_pred))
        self.acccuracy = accuracy_score(self.y_test, self.y_pred)
        self.prec = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        
        self.confusion = confusion_matrix(self.y_test,self.y_pred)
        self.report = classification_report(self.y_test, self.y_pred)
        
        self.evaluation = {'Accuracy': self.acccuracy, 'Precision':self.prec, 'recall':self.recall}
        pd.DataFrame.from_dict(self.evaluation, orient='index').to_csv(r'output\performance_decision_tree.csv',mode='a')
    
    def main(self):
        self.__train_test_split()
        self.__build_model()
        self.__obtain_metadata()
        self.__evaluate()
        
    
class KNeighbours:
    def __init__(self):
        self.df = dataset
        self.feature_list = list(self.df.columns)
        self.feature_list.remove('diagnosis')
        self.y = self.df.loc[:,'diagnosis']
        self.df.drop('diagnosis', inplace=True, axis=1)
        self.X = self.df.iloc[:,:]
    
    def __test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=21)
    
    def __build_model(self):
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(self.X_train, self.y_train)
    
    def __predict(self):
        return self.classifier.predict(self.X_test)
    
    def __evaluate(self):
        self.y_pred = self.__predict()
        self.acccuracy = accuracy_score(self.y_test, self.y_pred)
        self.prec = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        
        self.confusion = confusion_matrix(self.y_test,self.y_pred)
        self.report = classification_report(self.y_test, self.y_pred)
        
        self.evaluation = {'Accuracy': self.acccuracy, 'Precision':self.prec, 'recall':self.recall}
        pd.DataFrame.from_dict(self.evaluation, orient='index').to_csv(r'output\performance_KNN.csv',mode='a')
    
    def __plot(self):
        fig,axes = plt.subplots(2,10,figsize=(25,8), squeeze=False)
        fig.suptitle('Plotting Predictions')
        for i in range(10):
            feature = self.feature_list[i]
            x = self.X_train[feature]
            y = self.X_test[feature]
            sns.scatterplot(x=x, y=y, hue=self.y)
                
            
        plt.tight_layout()
        plt.savefig(r'plots\knn.png')
        
    
    def main(self):
        self.__test_train_split()
        self.__build_model()
        self.__evaluate()
        #self.__plot()

if __name__ == '__main__':
    dt = DecisionTree(reduced=True)
    kn = KNeighbours()
    dt.main()
    kn.main()