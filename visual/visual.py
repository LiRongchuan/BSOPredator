import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Plot:
    def __init__(self, grid):
        self.grid = grid
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, grid)
        self.ax.set_ylim(0, grid)
        self.scatter_virtuals = self.ax.scatter([], [], c='lightgreen', marker='o', label='Virtual Robots', s=10)
        self.scatter_robots = self.ax.scatter([], [], c='blue', marker='s', label='Real Robots', s=20)
        self.scatter_prey = self.ax.scatter([], [], c='red', marker='*', label='Prey', s=50)
        self.frame_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
        self.ax.legend(loc='upper right')
        self.frame_dir = 'temp'
        self.cnt = 0
        os.makedirs(self.frame_dir, exist_ok=True)
    
    def update(self, p_robots, preys):
        self.Ns = p_robots.shape[0]
        self.scatter_virtuals.set_offsets(p_robots[:, 1:].reshape(-1, 2))
        self.scatter_robots.set_offsets(p_robots[:, 0])
        self.scatter_prey.set_offsets(preys)
        frame_path = os.path.join(self.frame_dir, f'frame_{self.cnt:03d}.png')
        plt.savefig(frame_path)
        self.cnt += 1
        
    def save(self, fps=3, file=None):
        plt.close(self.fig)
        fig, ax = plt.subplots()
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        frames = []
        for i in range(self.cnt):
            img = plt.imread(os.path.join(self.frame_dir, f'frame_{i:03d}.png'))
            frame = ax.imshow(img, animated=True)
            frames.append([frame])
        for i in range(5):
            frames.append([frame])
        ani = animation.ArtistAnimation(fig, frames, interval=100)
        ani_file = f'output\\{self.grid}_{self.Ns}.gif' if file is None else 'output\\' + file
        ani.save(ani_file, writer='pillow', fps=fps)
        shutil.rmtree(self.frame_dir)
        plt.close(fig)