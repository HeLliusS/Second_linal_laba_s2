import os
import pandas as pd
import matplotlib.pyplot as plt

results_dir = "./cmake-build-debug/experiments_results"
for subdir in os.listdir(results_dir):
    if subdir.startswith("lr_"):
        path = os.path.join(results_dir, subdir, "loss.csv")
        df = pd.read_csv(path)
        plt.plot(df['epoch'], df['val_loss'], label=subdir)
plt.legend(); plt.title("Validation loss for different LR"); plt.show()
