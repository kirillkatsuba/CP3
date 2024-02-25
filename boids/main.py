import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
from funcs import  boids_agent, obstacle

app.use_app('pyqt6')

w, h = 4560, 1920
N = 1000
n = 10
dt = 0.1
asp = w / h
perception = 1/20
# walls_order = 8
better_walls_w = 0.05
vrange=(0, 0.1)
arange=(0, 0.05)

#                    c      a    s      w
coeffs = np.array([0.05, 0.02,   0.1,  0.03])

# 0  1   2   3   4   5
# x, y, vx, vy, ax, ay
# boids = np.zeros((N, 6), dtype=np.float64)
# init_boids(boids, asp, vrange=vrange)
# boids[:, 4:6] = 0.1

boids_first = boids_agent(N, asp, vrange, arange, perception)
obstacles = obstacle(n, asp, 0)

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=boids_first.directions(dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)
for i in range(n):
    circle = scene.visuals.Ellipse(center=(obstacles.obstacle_point[i, 0], obstacles.obstacle_point[i, 1]), radius=1/30, color=None, border_color='red')
    view.add(circle)

def update(event):
    boids_first.flocking(coeffs, better_walls_w)
    obstacles.separation(boids_first)
    boids_first.propagate(dt)
    arrows.set_data(arrows=boids_first.directions(dt))
    canvas.update()

if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
