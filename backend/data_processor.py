import pandas as pd


def load_data(file):
    df = pd.read_csv(file)
    return df


def get_basic_info(df):
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }

    return info


def get_column_info(df):
    column_info = []

    for column in df.columns:
        column_info.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Missing Values": int(df[column].isnull().sum()),
            "Unique Values": int(df[column].nunique())
        })

    return pd.DataFrame(column_info)