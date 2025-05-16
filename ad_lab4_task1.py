import pandas as pd
import numpy as np
import timeit
import os


df = pd.read_csv(
    os.path.join(os.path.dirname(__file__),"household_power_consumption.txt"),
    sep=";",
    na_values="?",
    dtype=str
)

df.dropna(inplace=True)

df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)

numeric_cols = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]
df[numeric_cols] = df[numeric_cols].astype(float)
data_np = df[numeric_cols].to_numpy()
datetime_np = df["DateTime"].to_numpy()


idx = {col: i for i, col in enumerate(numeric_cols)}


#1\ загальна споживана потужність (>5кВт)
def pandas_1():
    return df[df["Global_active_power"] > 5]

def numpy_1():
    return data_np[data_np[:, idx["Global_active_power"]] > 5]


#2\ вольтаж (>235 В)
def pandas_2():
    return df[df["Voltage"] > 235]

def numpy_2():
    return data_np[data_np[:, idx["Voltage"]] > 235]


#3\ сила струму (в межах 19-20A) + пральна машина та холодильник > бойлер та кондиціонер
def pandas_3():
    mask = (df["Global_intensity"] >= 19) & (df["Global_intensity"] <= 20)
    sub = df[mask]
    return sub[sub["Sub_metering_2"] > sub["Sub_metering_3"]]

def numpy_3():
    mask = (data_np[:, idx["Global_intensity"]] >= 19) & \
           (data_np[:, idx["Global_intensity"]] <= 20)
    sub = data_np[mask]
    return sub[sub[:, idx["Sub_metering_2"]] > sub[:, idx["Sub_metering_3"]]]


#4\ 500000 записів випадковим чином -> для них середні величини всіх груп
def pandas_4():
    sample = df.sample(n=500000, random_state=24)
    return sample[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].mean()

def numpy_4():
    np.random.seed(24)
    indices = np.random.choice(len(data_np), size=500000, replace=False)
    sample = data_np[indices]
    return sample[:, [idx["Sub_metering_1"], idx["Sub_metering_2"], idx["Sub_metering_3"]]].mean(axis=0)



#5\ після 18:00 споживання > 6Квт -> основне споживання: пральна машина, сушарка, холодильник, освітлення
def pandas_5():
    after_6pm = df[df["DateTime"].dt.hour >= 18]
    high_power = after_6pm[after_6pm["Global_active_power"] > 6]
    group2_dominant = high_power[
        (high_power["Sub_metering_2"] > high_power["Sub_metering_1"]) &
        (high_power["Sub_metering_2"] > high_power["Sub_metering_3"])
    ]
    mid = len(group2_dominant) // 2
    first_half = group2_dominant.iloc[:mid].iloc[::3]
    second_half = group2_dominant.iloc[mid:].iloc[::4]
    return pd.concat([first_half, second_half])


def numpy_5():
    hours = pd.to_datetime(datetime_np).hour
    after_6pm_mask = hours >= 18
    high_power_mask = data_np[:, idx["Global_active_power"]] > 6
    combined_mask = after_6pm_mask & high_power_mask

    selected = data_np[combined_mask]
    dt_selected = datetime_np[combined_mask]

    group2 = selected[:, idx["Sub_metering_2"]]
    group1 = selected[:, idx["Sub_metering_1"]]
    group3 = selected[:, idx["Sub_metering_3"]]

    dominant_mask = (group2 > group1) & (group2 > group3)
    final = dt_selected[dominant_mask]

    mid = len(final) // 2
    first_half = final[:mid][::3]
    second_half = final[mid:][::4]
    return np.concatenate([first_half, second_half], axis=0)



# timeit
functions = [
    ("1 Pandas", "pandas_1()"),
    ("1 NumPy", "numpy_1()"),

    ("2 Pandas", "pandas_2()"),
    ("2 NumPy", "numpy_2()"),

    ("3 Pandas", "pandas_3()"),
    ("3 NumPy", "numpy_3()"),

    ("4 Pandas", "pandas_4()"),
    ("4 NumPy", "numpy_4()"),

    ("5 Pandas", "pandas_5()"),
    ("5 NumPy", "numpy_5()"),
]

for name, stmt in functions:
    t = timeit.timeit(stmt, globals=globals(), number=3)
    print(f"since start of {name}: {t:.5f} s")

    if "NumPy" in name:
        result = eval(stmt)
        print(f"result for {name}:")

        if isinstance(result, pd.DataFrame):
            print(result.head(5))  
            print(f"lines in total: {len(result)}")
        elif isinstance(result, pd.Series):
            print(result)
        else:
            print(result)

        print("-" * 50)