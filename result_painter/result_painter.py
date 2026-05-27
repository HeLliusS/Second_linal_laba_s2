import pandas as pd
import matplotlib.pyplot as plt

path = "./cmake-build-debug/results/"

# График потерь
loss = pd.read_csv(path + 'loss.csv')
plt.figure()
plt.plot(loss['epoch'], loss['train_loss'], label='Train loss')
plt.plot(loss['epoch'], loss['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy')
plt.legend()
plt.title('Loss curves')
plt.savefig('loss_plot.png')
plt.show()

# Точки и разделяющая прямая
points = pd.read_csv(path + 'points.csv')
line = pd.read_csv(path + 'line.csv')
plt.figure()
plt.scatter(points[points['label']==0]['x1'], points[points['label']==0]['x2'], label='Class 0')
plt.scatter(points[points['label']==1]['x1'], points[points['label']==1]['x2'], label='Class 1')
plt.plot(line['x1'], line['x2'], 'r-', label='Decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Data and decision boundary')
plt.savefig('boundary.png')
plt.show()