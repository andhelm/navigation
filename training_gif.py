# training_gif.py

import imageio
import os

save_dir = "epoch_plots"
images = []
for file_name in sorted(os.listdir(save_dir)):
    if file_name.endswith(".png"):
        file_path = os.path.join(save_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('training_progress.gif', images, duration=0.5)
