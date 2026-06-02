import pandas as pd


def load_dataset(path):
    """
    Load CSV dataset from the given path.
    """
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully.")
        print("Shape:", df.shape)
        return df
    except FileNotFoundError:
        print("Dataset file not found. Please check the file path.")
        return None


def show_basic_info(df):
    """
    Display basic dataset information.
    """
    print("Dataset Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum().sort_values(ascending=False))


def clean_rain_dataset(df):
    """
    Clean the weatherAUS dataset for rain prediction.
    """

    df = df.copy()

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Extract useful date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Drop Date column
    df = df.drop("Date", axis=1)

    # Drop columns with many missing values
    columns_to_drop = ["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Remove rows where target value is missing
    df = df.dropna(subset=["RainTomorrow"])

    # Convert target column to numeric
    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    return df