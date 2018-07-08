import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from main import load_obj, all_models
import os

data = load_obj()
points = data['points']
limit = -points[0][0]

def create_video(model):
    filename = os.path.join('videos', '{}.mp4'.format(model.name))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    keys = [key for key in data.keys() if key.startswith(model.name)]

    def animate(i):
        ax.clear()
        ax.set_xlim(-limit,limit)
        ax.set_ylim(-limit,limit)
        labels = data[keys[i]]
        stacked = np.hstack([points, labels])
        ax.scatter(stacked[:,0], stacked[:,1], c=stacked[:,2], s=1, cmap='viridis')

    ani = animation.FuncAnimation(fig, animate, frames=len(keys))
    ani.save(filename, fps=25, extra_args=['-vcodec', 'libx264'])
