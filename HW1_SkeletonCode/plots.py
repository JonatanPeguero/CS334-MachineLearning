#THIS HOMEWORK IS MY OWN WORK, WRITTEN WITHOUT COPYING FROM OTHER STUDENTS
#OR DIRECTLY FROM LARGE LANGUAGE MODELS SUCH AS CHATGPT.
#Any collaboration or external resources have been properly acknowledged.
#Jonatan Peguero
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

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

def drop_na_numeric(inputdf):
    numeric_cols = inputdf.select_dtypes(include=[np.number]).columns
    initial_rows = inputdf.shape[0]
    outputdf = inputdf.dropna(subset=numeric_cols).reset_index(drop=True)
    removed_rows = initial_rows - outputdf.shape[0]
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with NaN in numeric columns.")
    else:
        print("No rows with NaN found in numeric columns.")
    return outputdf


def main():
    # Load data
    df_original = load_csv("data/penguins.csv")
    if df_original is None:
        return  # Exit if data loading failed

    # Remove rows with NaN in 'species' (if any)
    df_cleaned_species = remove_na(df_original, "species")

    # Drop rows with NaN in any numeric features
    df_cleaned = drop_na_numeric(df_cleaned_species)

    # Define numeric features and species
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    species = 'species'

    # Define color palette for species
    palette = {'Adelie': 'blue', 'Chinstrap': 'green', 'Gentoo': 'red'}

    # Generate all unique pairs of numeric features
    feature_pairs = list(combinations(numeric_features, 2))

    # Iterate over each feature pair to create scatterplots
    for (x_feature, y_feature) in feature_pairs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_cleaned, x=x_feature, y=y_feature, hue=species, palette=palette, style=species)
        plt.title(f'{y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()} by Species')
        plt.xlabel(x_feature.replace("_", " ").title())
        plt.ylabel(y_feature.replace("_", " ").title())
        plt.legend(title='Species')
        plt.tight_layout()
        plt.savefig(f'scatterplot_{x_feature}_vs_{y_feature}.png')  # Save the plot
        plt.show()
if __name__ == "__main__":
    main()
