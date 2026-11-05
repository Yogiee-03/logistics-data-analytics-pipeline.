import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Step 1: Load and Inspect Dataset
def load_and_inspect(filepath):
    df = pd.read_csv(filepath)
    print("Initial Dataset Shape:", df.shape)
    return df

# Step 2: Handle Structural Inconsistencies & Duplicates
def clean_structure(df):
    df.drop_duplicates(subset=["Shipment_ID"], keep="last", inplace=True)
    df["Shipment_Status"] = df["Shipment_Status"].astype(str).str.strip().str.upper()
    return df

# Step 3: Parse Timestamps & Enforce Hard Logical Constraints
def validate_and_parse_timestamps(df):
    df["Dispatch_Timestamp"] = pd.to_datetime(df["Dispatch_Timestamp"], errors="coerce")
    df["Delivery_Timestamp"] = pd.to_datetime(df["Delivery_Timestamp"], errors="coerce")
    
    # Calculate Actual Lead Time in Hours
    df["Actual_Lead_Time_Hrs"] = (
        df["Delivery_Timestamp"] - df["Dispatch_Timestamp"]
    ).dt.total_seconds() / 3600.0

    # Drop physically impossible rows (Negative lead times or zero/negative weight)
    df = df[
        (df["Actual_Lead_Time_Hrs"] > 0)
        & (df["Distance_KM"] > 0)
        & (df["Package_Weight_KG"] > 0)
    ]
    return df

# Step 4: Handle Missing Values
def impute_missing_values(df):
    # Median Imputation by Route Corridor for Traffic Index
    df["Traffic_Delay_Index"] = df.groupby("Route_ID")[
        "Traffic_Delay_Index"
    ].transform(lambda x: x.fillna(x.median()))

    # Fallback global median if entire route corridor was null
    df["Traffic_Delay_Index"].fillna(df["Traffic_Delay_Index"].median(), inplace=True)
    return df

# Step 5: Outlier Treatment via IQR Capping (Winsorization)
def cap_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

def treat_outliers(df):
    df["Distance_KM_Cleaned"] = cap_outliers_iqr(df["Distance_KM"])
    df["Freight_Cost_Cleaned"] = cap_outliers_iqr(df["Freight_Cost_USD"])
    return df

# Step 6: Feature Scaling & Normalization
def scale_features(df):
    # Log transformation for highly skewed financial costs
    df["Log_Freight_Cost"] = np.log1p(df["Freight_Cost_Cleaned"])

    # Standard Scaling for distance and weight (Mean=0, Std=1)
    scaler = StandardScaler()
    df[["Distance_KM_Scaled", "Weight_KG_Scaled"]] = scaler.fit_transform(
        df[["Distance_KM_Cleaned", "Package_Weight_KG"]]
    )

    # Min-Max Scaling for Traffic Index (0 to 1 range)
    minmax_scaler = MinMaxScaler()
    df["Traffic_Index_Normalized"] = minmax_scaler.fit_transform(
        df[["Traffic_Delay_Index"]]
    )
    return df

if __name__ == "__main__":
    print("Logistics Data Preprocessing Pipeline Script Loaded Successfully.")
