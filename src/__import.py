import pandas as pd


class Import:
    def __init__(self):
        self.df = pd.read_csv(r'datasets\breast-cancer.csv')
    
    def __return_dataset(self):
        return self.df

    def main(self):
        return self.__return_dataset()
    
if __name__ == '__main__':
    pass