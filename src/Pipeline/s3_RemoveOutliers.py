import os
import pandas as pd
import yaml

class RemoveOutlier:
    def load_yaml(self, yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            return data.get('airlineName', {})

    def remove_outliers(self, df, airlineName):
        cleaned_df = pd.DataFrame(columns=df.columns)
        
        for airline, quartiles in airlineName.items():
            airDataSet = df[df['Airline'] == airline]
            q1 = airDataSet['Price'].quantile(quartiles[0])
            q3 = airDataSet['Price'].quantile(quartiles[1])
            IQR = q3 - q1
            lowerLimit = q1 - IQR * 1.5
            upperLimit = q3 + IQR * 1.5
            
            lowerLimitIndex = airDataSet[airDataSet['Price'] <= lowerLimit].index
            upperLimitIndex = airDataSet[airDataSet['Price'] >= upperLimit].index
            
            if airDataSet.shape[0] > 5:
                airDataSet = airDataSet.drop(lowerLimitIndex).drop(upperLimitIndex)
            
            cleaned_df = pd.concat([cleaned_df, airDataSet], axis=0)
        
        return cleaned_df

    def save_file(self, df, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    file_path = "Data/02_CleanedData/CleanedData.csv"
    yaml_path = "constants.yaml"
    noOutlierDataFilePath = "Data/03_noOutlierData/noOutlierDataFile.csv"
    
    df = pd.read_csv(file_path)
    remover = RemoveOutlier()
    airlineName = remover.load_yaml(yaml_path)
    cleaned_df = remover.remove_outliers(df, airlineName)
    remover.save_file(cleaned_df, noOutlierDataFilePath)
