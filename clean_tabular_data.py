def clean_product_data(df):
    """Drops null values and formats the price column to float64
    
    Parameters:
    df (Pandas Dataframe): A pandas dataframe containing product data.

    Returns:
    Pandas Dataframe: df
      
    """
    df = df.dropna()
    df["price"] = df["price"].str.replace("£","").str.replace(",","")
    df["price"] = df["price"].astype("float64")
    return df
    
    