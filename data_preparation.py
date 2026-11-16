import os
import pandas as pd
import matplotlib.pyplot as plt
"""
#Folder paths

xray_folder = "data/xrays"
mask_folder = "data/masks"
csv_folder = "data/CSVs"

#Go through every x-ray and check if it has a mask

with_mask = []
without_mask = []

for file_name in os.listdir(xray_folder):
    xray_path = os.path.join(xray_folder, file_name)
    mask_path = os.path.join(mask_folder, file_name)

    if os.path.exists(mask_path):
        with_mask.append([xray_path, mask_path])
    else:
        without_mask.append([xray_path])

#Make tables (DataFrames)
df = pd.DataFrame(with_mask, columns=["xray", "mask"])
df_no_mask = pd.DataFrame(without_mask, columns=["xray"])

#Save the CSV files
df.to_csv(os.path.join(csv_folder, "data.csv"), index=False)
df_no_mask.to_csv(os.path.join(csv_folder, "no_mask.csv"), index=False)
"""
# mitra
data_list = []
xray_dir = "data/xrays"

for xray in os.listdir(xray_dir):
    xray_path = os.path.join(xray_dir, xray)
    mask_path = xray_path.replace("xrays", "masks")
    if not os.path.exists(mask_path):
        mask_path = None

    data_list.append((xray_path, mask_path))
data_df = pd.DataFrame(data_list, columns=["xrays", "masks"])
data_df.to_csv("data/CSVs/dataset.csv", index=False)

#create train, val and test csv files

test_df = data_df[data_df["masks"].isna()]
labeled_df = data_df[data_df["masks"].notna()]

train_df = labeled_df.sample(frac=0.8, random_state=42)
val_df = labeled_df.drop(train_df.index)

train_df.to_csv("data/CSVs/train.csv", index=False)
val_df.to_csv("data/CSVs/val.csv", index=False)
test_df.to_csv("data/CSVs/test.csv", index=False)


#bar chart to check the csv files
names = ["dataset", "train", "val", "test"]
counts = [len(data_df), len(train_df), len(val_df), len(test_df)]

plt.bar(names, counts)
plt.title("Distribution of datasets")
plt.ylabel("Number of samples")
plt.savefig("data/CSVs/bar_chart.png")
plt.show()