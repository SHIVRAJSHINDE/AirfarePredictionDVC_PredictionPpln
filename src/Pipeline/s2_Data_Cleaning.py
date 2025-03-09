import pandas as pd
import os

class DataCleaningClass:
    def read_csv(self, file_path: str) -> pd.DataFrame:
        """Reads a CSV file and returns a DataFrame."""
        df = pd.read_csv(file_path)
        return df

    def clean_total_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values in 'Total_Stops' column with the mode."""
        mode_of_total_stops = df['Total_Stops'].mode()[0]
        df.fillna({'Total_Stops': mode_of_total_stops}, inplace=True)
        print("Missing values in 'Total_Stops' filled with mode.")
        return df

    def clean_airline_column(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df)
        """Cleans the 'Airline' column by replacing specific values."""
        df['Airline'].replace("Multiple carriers Premium economy", "Multiple carriers")
        df['Airline'].replace("Jet Airways Business", "Jet Airways")
        df['Airline'].replace("Vistara Premium economy", "Vistara")
        print("Airline names cleaned.")
        return df

    def clean_destination_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces 'New Delhi' with 'Delhi' in the 'Destination' column."""
        print(df)
        df['Destination'].replace(to_replace="New Delhi", value="Delhi", inplace=True)
        print("'New Delhi' replaced with 'Delhi' in 'Destination'.")
        return df

    def create_duration_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a new column 'hoursMinutes' to represent flight duration in minutes."""
        df["hoursMinutes"] = 0
        for i in df.index:
            if " " in df.loc[i, 'Duration']:
                column1 = df.loc[i, 'Duration'].split(" ")[0]
                column2 = df.loc[i, 'Duration'].split(" ")[1]

                if "h" in column1:
                    column1 = (int(column1.replace("h", "")) * 60)
                elif "m" in column1:
                    column1 = (int(column1.replace("m", "")))

                if "h" in column2:
                    column2 = (int(column2.replace("h", "")) * 60)
                elif "m" in column2:
                    column2 = (int(column2.replace("m", "")))

                df.loc[i, 'hoursMinutes'] = column1 + column2
            else:
                column1 = df.loc[i, 'Duration']

                if "h" in column1:
                    column1 = (int(column1.replace("h", "")) * 60)
                elif "m" in column1:
                    column1 = (int(column1.replace("m", "")))

                df.loc[i, 'hoursMinutes'] = column1

        print("'hoursMinutes' column created from 'Duration'.")
        return df

    def process_date_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts and processes 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' columns."""
        print(df)

        df['Day'] = pd.to_datetime(df["Date_of_Journey"], format="%d-%m-%Y").dt.day
    
        df['Month'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.month
     
        df['Year'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.year
        
        df['Dept_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
        
        df['Dept_Minute'] = pd.to_datetime(df['Dep_Time']).dt.minute

        #df['Arr_Hour'] = pd.to_datetime(df['Arrival_Time'], format="%d-%m-%Y %H:%M").dt.hour
        #df['Arr_Minute'] = pd.to_datetime(df['Arrival_Time'],format="%d-%m-%Y %H:%M").dt.minute

        print("Date and time columns processed.")
        return df

    def arrival_Hours_Minutes(self,df:pd.DataFrame)-> pd.DataFrame:
        def extract_hour(value):
            try:
                return pd.to_datetime(value, format="%d-%m-%Y %H:%M").hour  # Full datetime format
            except ValueError:
                return pd.to_datetime(value, format="%H:%M").hour  # Time-only format
        
        df['Arr_Hour'] = df['Arrival_Time'].apply(extract_hour)

        def extract_minute(value):
            try:
                return pd.to_datetime(value, format="%d-%m-%Y %H:%M").minute  # Full datetime format
            except ValueError:
                return pd.to_datetime(value, format="%H:%M").minute  # Time-only format

        df['Arr_Minute'] = df['Arrival_Time'].apply(extract_minute)

        return df


    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unnecessary columns from the DataFrame."""
        df = df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info'], axis=1)
        print("Unnecessary columns dropped.")
        return df

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorders columns for the final DataFrame."""
        df = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Day', 'Month', 'Year', 
                 'Dept_Hour', 'Dept_Minute', 'Arr_Hour', 'Arr_Minute', 'hoursMinutes', 'Price']]
        print("Columns reordered.")
        return df

    def save_file(self, df: pd.DataFrame, directory: str, filename: str) -> None:
        """Saves the DataFrame to a CSV file in the specified directory."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' was created.")
        else:
            print(f"Directory '{directory}' already exists.")

        file_path = os.path.join(directory, filename)
        df.to_csv(file_path, index=False)
        print(f"File has been saved to {file_path}")


# Example usage:
if __name__ == "__main__":
    raw_file_path = "Data\\01_RawData\\Airline.csv"
    directory = "Data\\02_CleanedData\\"
    filename = "CleanedData.csv"

    # Create an instance of DataCleaningClass
    data_cleaning_obj = DataCleaningClass()
    df = data_cleaning_obj.read_csv(raw_file_path)
    # Apply the cleaning functions
    df = data_cleaning_obj.clean_total_stops(df)
    df = data_cleaning_obj.clean_airline_column(df)
    df = data_cleaning_obj.clean_destination_column(df)
    df = data_cleaning_obj.create_duration_column(df)
    df = data_cleaning_obj.process_date_time_columns(df)
    df = data_cleaning_obj.arrival_Hours_Minutes(df)
    df = data_cleaning_obj.drop_unnecessary_columns(df)
    df = data_cleaning_obj.reorder_columns(df)
        # Save the cleaned data
    data_cleaning_obj.save_file(df,directory,filename)
