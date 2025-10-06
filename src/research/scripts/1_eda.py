"""
The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

Categorical Features - Transaction
ProductCD
card1 - card6
addr1, addr2
P_emaildomain
R_emaildomain
M1 - M9
Categorical Features - Identity
DeviceType
DeviceInfo
id_12 - id_38
The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).
"""

"""
Skip EDA. Assume, that it's made by following:
[1] https://www.kaggle.com/code/cdeotte/eda-for-columns-v_columns-and-id#NAN-search
[2] https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600

Main idea of EDA [1]: count nans -> there are a lot of columns with the same nan counts -> 
-> calculate correlations for features inside groups with the same nan count ->
-> extract highly correlated subgroups (r > 0.75) -> save the features with lowest ids from each subgroup to
remove redundancy but not lost too many info
"""

# COLUMNS WITH STRINGS
string_columns = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]
string_columns += [
    "id-12",
    "id-15",
    "id-16",
    "id-23",
    "id-27",
    "id-28",
    "id-29",
    "id-30",
    "id-31",
    "id-33",
    "id-34",
    "id-35",
    "id-36",
    "id-37",
    "id-38",
]

# FIRST 53 COLUMNS
all_column_names = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

# V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
# https://www.kaggle.com/cdeotte/eda-for-columns-v_columns-and-id
v_columns = [1, 3, 4, 6, 8, 11]
v_columns += [13, 14, 17, 20, 23, 26, 27, 30]
v_columns += [36, 37, 40, 41, 44, 47, 48]
v_columns += [54, 56, 59, 62, 65, 67, 68, 70]
v_columns += [76, 78, 80, 82, 86, 88, 89, 91]

# v_columns += [96, 98, 99, 104] #relates to groups, no NAN
v_columns += [107, 108, 111, 115, 117, 120, 121, 123]  # maybe group, no NAN
v_columns += [124, 127, 129, 130, 136]  # relates to groups, no NAN

# LOTS OF NAN BELOW
v_columns += [138, 139, 142, 147, 156, 162]  # b1
v_columns += [165, 160, 166]  # b1
v_columns += [178, 176, 173, 182]  # b2
v_columns += [187, 203, 205, 207, 215]  # b2
v_columns += [169, 171, 175, 180, 185, 188, 198, 210, 209]  # b2
v_columns += [218, 223, 224, 226, 228, 229, 235]  # b3
v_columns += [240, 258, 257, 253, 252, 260, 261]  # b3
v_columns += [264, 266, 267, 274, 277]  # b3
v_columns += [220, 221, 234, 238, 250, 271]  # b3

v_columns += [294, 284, 285, 286, 291, 297]  # relates to grous, no NAN
v_columns += [303, 305, 307, 309, 310, 320]  # relates to groups, no NAN
v_columns += [281, 283, 289, 296, 301, 314]  # relates to groups, no NAN
# v_columns += [332, 325, 335, 338] # b4 lots NAN

all_column_names += ["V" + str(x) for x in v_columns]
all_column_names += (
    ["id_0" + str(x) for x in range(1, 10)]
    + ["id_" + str(x) for x in range(10, 34)]
    + ["id-0" + str(x) for x in range(1, 10)]
    + ["id-" + str(x) for x in range(10, 34)]
)

column_types = {}
for c in all_column_names:
    column_types[c] = "float32"
for c in string_columns:
    column_types[c] = "category"

"""
# just for fun:
import os
from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport

datasets_dir = Path("..", "..", "Datasets", "ieee-fraud-detection")
train_df = pd.read_csv(Path(datasets_dir, "train_transaction.csv"))
train_df = train_df.sample(n=int(0.05 * len(train_df)), random_state=45)

profiles_dir = Path("data", "interim")
profile = ProfileReport(train_df, title="Train Report")
profile.to_file(Path(profiles_dir, "train_profile.html"))
"""

print("done")
