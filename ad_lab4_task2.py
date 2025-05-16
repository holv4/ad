import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import timeit
import os

file_path = os.path.join(os.path.dirname(__file__), "chronic_kidney_disease.txt")

#1 Зчитування даних
def read_pandas(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    data_lines = [line.strip() for line in lines if not line.startswith('@') and line.strip()]
    data_clean = [line.rstrip(',').split(',') for line in data_lines if len(line.strip()) > 0]

    columns = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
               "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
               "htn", "dm", "cad", "appet", "pe", "ane", "class"]

    trimmed_data = [row[:25] if len(row) > 25 else row for row in data_clean]

    padded_data = [row + [np.nan] * (25 - len(row)) if len(row) < 25 else row for row in trimmed_data]

    df = pd.DataFrame(padded_data, columns=columns)
    df.replace('?', np.nan, inplace=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue
    return df

def read_numpy(path):
    df = read_pandas(path)
    num_df = df.select_dtypes(include=[np.number])
    return num_df.to_numpy()



#2 заповнення пропущених значень
def fill_missing_pandas(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def fill_missing_numpy(data):
    data = np.where(np.isnan(data), np.nanmedian(data, axis=0), data)
    return data


#3 нормалізація/стандартизація
def normalize_numpy(data):
    return (data - np.min(data, axis=0)) / (np.ptp(data, axis=0))

def standardize_numpy(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def normalize_pandas(df):
    return (df - df.min()) / (df.max() - df.min())

def standardize_pandas(df):
    return (df - df.mean()) / df.std()


#4 візуалізація
def plot_histogram(df):
    plt.hist(df['age'].dropna().astype(float), bins=10, edgecolor='black')
    plt.title("Гістограма віку (age)")
    plt.xlabel("Вік")
    plt.ylabel("Кількість")
    plt.savefig("output1.png")


def plot_scatter(df):
    df_numeric = df[["bgr", "sc"]].dropna()
    plt.scatter(df_numeric["bgr"].astype(float), df_numeric["sc"].astype(float), alpha=0.7)
    plt.title("Залежність Serum Creatinine від Blood Glucose")
    plt.xlabel("Blood Glucose (bgr)")
    plt.ylabel("Serum Creatinine (sc)")
    plt.grid(True)
    plt.savefig("output2.png")



#5 коефіцієнти кореляції
def correlation_coeffs(df):
    df_corr = df[["bgr", "sc"]].dropna()
    bgr = df_corr["bgr"].astype(float)
    sc = df_corr["sc"].astype(float)
    pearson_corr, _ = pearsonr(bgr, sc)
    spearman_corr, _ = spearmanr(bgr, sc)
    print(f"Коефіцієнт Пірсона: {pearson_corr:.4f}")
    print(f"Коефіцієнт Спірмена: {spearman_corr:.4f}")


#6 оne-hot encoding
def one_hot_encode(df):
    encoded = pd.get_dummies(df, columns=["class"], prefix="class")
    print("One-Hot Encoding для class:")
    print(encoded[["class_ckd", "class_notckd"]].head(10))
    return encoded


#7 pairplot
def show_pairplot(df):
    numeric_cols = ["age", "bp", "bgr", "bu", "sc"]
    df_sub = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
    sns.pairplot(df_sub)
    plt.suptitle("Pairplot числових атрибутів", y=1.02)
    plt.savefig("output3.png")


#8 обчислення часу виконання
def pandas_pipeline():
    df = read_pandas(file_path)
    df = fill_missing_pandas(df)
    norm = normalize_pandas(df.select_dtypes(include=[np.number]))
    std = standardize_pandas(df.select_dtypes(include=[np.number]))

def numpy_pipeline():
    data = read_numpy(file_path)
    data = fill_missing_numpy(data)
    norm = normalize_numpy(data)
    std = standardize_numpy(data)




df_pandas = read_pandas(file_path)
df_pandas = fill_missing_pandas(df_pandas)
data_numpy = read_numpy(file_path)
data_numpy = fill_missing_numpy(data_numpy)

plot_histogram(df_pandas)
plot_scatter(df_pandas)
correlation_coeffs(df_pandas)
one_hot_encode(df_pandas)
show_pairplot(df_pandas)

pandas_time = timeit.timeit(pandas_pipeline, number=100)
numpy_time = timeit.timeit(numpy_pipeline, number=100)
print(f"\nPandas: час обробки за 100 повторів: {pandas_time:.4f} сек")
print(f"NumPy: час обробки за 100 повторів: {numpy_time:.4f} сек")
