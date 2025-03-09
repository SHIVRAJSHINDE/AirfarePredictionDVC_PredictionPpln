import os
import pandas as pd

class DataIngestionClass:
    def read_csv(source_path):
        # Read the CSV file and return the DataFrame
        df = pd.read_csv(source_path)
        print(df)
        return df

    def save_file(df, directory, filename):
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' was created.")
        else:
            print(f"Directory '{directory}' already exists.")

        # Construct the file path
        file_path = os.path.join(directory, filename)
        
        # Save the DataFrame to the file
        df.to_csv(file_path, index=False)  # index=False to avoid writing row indices
        print(f"File has been saved to {file_path}")

# This block will only execute if this script is run directly
if __name__ == "__main__":
    source_path = 'C:\\Users\\SHIVRAJ SHINDE\\JupiterWorking\\XL_ML\\Z_DataSets\\01_AirlineData\\Airline.csv'
    directory = "Data/01_RawData/"
    filename = "Airline.csv"

    df = DataIngestionClass.read_csv(source_path)  # Read the CSV file
    DataIngestionClass.save_file(df, directory, filename)  # Save the DataFrame to the destination
