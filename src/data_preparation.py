import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_table

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"


def perform_eda(df):

    print("\n" + "="*50)
    print("DATA PREPARATION & EDA")
    print("="*50)
    
    n_samples, n_features = df.shape
    print(f"Number of Samples: {n_samples}")
    print(f"Number of Features: {n_features}")
    
    missing_values = df.isnull().sum().sum()
    print(f"Missing Values: {missing_values}")
    if missing_values > 0:
        print("Detailed missing values per column:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        
    if "Class" in df.columns:
        class_dist = df["Class"].value_counts()
        legit_count = class_dist.get(0, 0)
        fraud_count = class_dist.get(1, 0)
        fraud_ratio = (fraud_count / n_samples) * 100
        
        print("\nClass Distribution:")
        print(f"  Legitimate (0): {legit_count}")
        print(f"  Fraudulent (1): {fraud_count}")
        print(f"  Fraud Ratio:    ~{fraud_ratio:.3f}%")
        
    print("="*50 + "\n")
    

    print("Generating descriptive statistics...")
    stats_df = df.describe().T.reset_index().rename(columns={'index': 'Feature'})
    
    save_table(stats_df, name="descriptive_statistics", sub="tables")
    
    return stats_df


def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    
    print(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    perform_eda(df)

    initial_shape = df.shape
    df = df.drop_duplicates()
    duplicates_removed = initial_shape[0] - df.shape[0]
    print(f"\nRemoved {duplicates_removed} duplicate rows. Remaining rows: {df.shape[0]}")
    
    return df


def split_and_save_data(df, test_size=0.2, random_state=42):
    print("Performing stratified train-test split (80/20)...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Class"],
        random_state=random_state
    )
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    print(f"Train dataset saved to: {TRAIN_DATA_PATH} (Shape: {train_df.shape})")
    print(f"Test dataset saved to:  {TEST_DATA_PATH} (Shape: {test_df.shape})")


if __name__ == "__main__":
    df = load_and_clean_data(RAW_DATA_PATH)
    split_and_save_data(df)