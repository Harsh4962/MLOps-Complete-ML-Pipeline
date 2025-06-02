import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Ensure the "logs" directory exists
"""
We first create a folder named 'logs' using os, where our log files
for each component will be saved.
"""
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
"""
We create a logger object named 'data_ingestion' and
set its level to 'DEBUG'.
"""
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

"""
We create a StreamHandler to show the logs in our terminal and 
set its level to 'DEBUG'.
"""
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

"""
We create a FileHandler. First, we define the path to the log file
'named data_ingestion.log' inside our 'logs' directory,
then configure it and set its level to 'DEBUG'.
"""
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Setting format
"""
We define the format in which we want to display our log messages.
Used here: 'time' - 'logger name' - 'log level' - 'actual log message'.
"""
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

"""
After creating our handlers and setting their formats,
we add them to the logger object.
"""
logger.addHandler(console_handler)
logger.addHandler(file_handler)

################ LOADING DATA ################
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        # Try to read the CSV file from the given URL or file path
        df = pd.read_csv(data_url)
        # Log a debug message saying data was loaded successfully from the source
        logger.debug('Data loaded from %s', data_url)
        return df

    except pd.errors.ParserError as e:
        # If there's a parsing error (e.g., malformed CSV), catch it and assign to variable 'e'
        # Log the error message with details of the exception 'e'
        logger.error('Failed to parse the CSV file : %s', e)
        # Re-raise the same exception so the caller knows something went wrong
        raise

    except Exception as e:
        # Catch any other unexpected exceptions and assign to 'e'
        # Log the unexpected error details for debugging
        logger.error('Unexpected error occurred while loading the data : %s', e)
        # Re-raise the exception so the program can handle it at a higher level or stop
        raise

################ PREPROCESSING DATA ################
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by dropping unwanted columns and renaming others"""
    try:
        # Drop unnecessary columns by their names, modify df in-place
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        # Rename columns 'v1' to 'target' and 'v2' to 'text' in-place
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        # Log a debug message indicating preprocessing was successful
        logger.debug('Data preprocessing completed')
        # Return the modified DataFrame
        return df

    except KeyError as e:
        # If one of the columns to drop or rename is missing, catch KeyError and log it
        logger.error('Missing column in the dataframe: %s', e)
        # Re-raise the exception to notify the caller about this error
        raise

    except Exception as e:
        # Catch any other unexpected errors during preprocessing
        logger.error("Unexpected error during preprocessing: %s", e)
        # Re-raise the exception for proper handling upstream
        raise

################ SAVING DATA ################
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets as CSV files in the specified folder"""
    try:
        # Create a 'raw' folder inside the given data_path to store raw data files
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)  # create folder if not exists
        # Save the training data as 'train.csv' without row indices
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        # Save the test data as 'test.csv' without row indices
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        # Log a debug message to confirm data was saved successfully
        logger.debug('Train and Test data saved to %s', raw_data_path)

    except Exception as e:
        # If any unexpected error occurs during saving, log the error message
        logger.error('Unexpected error occurred while saving the data: %s', e)
        # Re-raise the exception to let the caller handle it
        raise

def main():
    try:
        test_size = 0.2  # Portion of data to keep for testing
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv' # URL or path to the data source

        # Load the data from CSV file or URL
        df = load_data(data_url=data_path)
        # Preprocess the loaded data (drop/rename columns, clean data)
        final_df = preprocess_data(df)
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        # Save the split datasets into the ./data/raw folder as CSV files
        save_data(train_data=train_data, test_data=test_data, data_path='./data')

    except Exception as e:
        # Log any errors that happen during the whole data ingestion pipeline
        logger.error('Failed to complete the data ingestion process : %s', e)
        # Also print the error to the console for immediate feedback
        print(f"Error : {e}")

# If this script is run directly, start execution from main()
if __name__ == '__main__':
    main()

