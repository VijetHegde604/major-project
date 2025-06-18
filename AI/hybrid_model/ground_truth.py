import os
import pandas as pd

test_dir = 'sample_data'  

entries = []
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        label = filename.split('_')[-1].split('.')[0]  # assumes `img1_label.jpg` format
        entries.append({'image_name': filename, 'true_label': label})

df = pd.DataFrame(entries)
df.to_csv("ground_truth.csv", index=False)
print("✅ Saved ground truth to ground_truth.csv")
