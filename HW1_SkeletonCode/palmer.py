#THIS HOMEWORK IS MY OWN WORK, WRITTEN WITHOUT COPYING FROM OTHER STUDENTS
#OR DIRECTLY FROM LARGE LANGUAGE MODELS SUCH AS CHATGPT.
#Any collaboration or external resources have been properly acknowledged.
#Jonatan Peguero
"""
Pandas DataFrame Manipulation with Palmer Penguins Dataset
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
import pandas as pd
import numpy as np 
def load_csv(inputfile):
    """
    Load the csv as a pandas data frame
    
    Parameters
    ----------
    inputfile : string
        filename of the csv to load

    Returns
    -------
    csvdf : pandas.DataFrame
        return the pandas dataframe with the contents
        from the csv inputfile
    """
    # TODO: Implement this function
    csvdf = pd.read_csv(inputfile)
    return csvdf


def remove_na(inputdf, colname):
    """
    Remove the rows in the dataframe with NA as values 
    in the column specified.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to check and remove rows with NA

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    # TODO: Implement this function
    if colname not in inputdf.columns:
        return inputdf.copy()
    outputdf = inputdf.dropna(subset=[colname]).reset_index(drop=True)
    return outputdf


def onehot(inputdf, colname):
    """
    Convert the column in the dataframe into a one hot encoding.
    The newly converted columns should be at the end of the data
    frame and you should also drop the original column.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to one-hot encode

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    # TODO: Implement this function
    if colname not in inputdf.columns:
        return inputdf.copy()
    
    one_hot = pd.get_dummies(inputdf[colname], prefix=colname, dtype=int)
    outputdf = pd.concat([inputdf, one_hot], axis=1)
    outputdf = outputdf.drop(columns=[colname]).reset_index(drop=True)
    return outputdf


def to_numeric(inputdf):
    """
    Extract all the 
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe

    Returns
    -------
    outputnp : numpy.ndarray
        return the numeric contents of the input dataframe as a 
        numpy array
    """
    # TODO: Implement this function
    if inputdf is None: 
        return None 
    numeric_df = inputdf.select_dtypes(include=[np.number])
    print(numeric_df.columns)
    print(numeric_df.dtypes)
    print(f"{numeric_df.shape}")
    
    outputnp = numeric_df.to_numpy()
    
    return outputnp


def main():
    # Load data
    df = load_csv("data/penguins.csv")
    print(df)
    # Remove NA
    df = remove_na(df, "species")
    print(df)

    # One hot encoding
    df = onehot(df, "species")
    print(df)

    # Convert to numeric
    df_np = to_numeric(df)
    print(df_np)

if __name__ == "__main__":
    main()
