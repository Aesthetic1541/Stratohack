import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df

def encode_categorical_columns(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col + "_code"] = le.fit_transform(df[col])
    return df

def create_interaction_feature(df, col1, col2, new_col):
    combined = df[col1].astype(str) + "_" + df[col2].astype(str)
    le = LabelEncoder()
    df[new_col] = le.fit_transform(combined)
    return df

def save_feature_importance_plot(model, feature_names, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.barplot(x=model.feature_importances_, y=feature_names)
    plt.title("Feature Importances")
    plt.savefig(filename)
    plt.close()
