import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
from funcs import  boids_agent, obstacle

app.use_app('pyqt6')

# инициализруем необходимые параметры
w, h = 2560, 1100
N = 1000 # агенты
n = 7 # препятствия
dt = 0.1
asp = w / h
perception = 1/20
better_walls_w = 0.05
vrange=(0, 0.1)
arange=(0, 0.05)

coeffs = np.array([0.2, 0.02,   0.15,  0.1, 0.1, 0.001])

boids_first = boids_agent(N, asp, vrange, arange, perception)
obstacles = obstacle(n, asp)

# создаем сцену, отрисовываем агентов и препятствия
canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(-0.2, -0.2, 1.4*asp, 1.8))
arrows = scene.Arrow(arrows=boids_first.directions(boids_first.boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)
for i in range(n):
    circle = scene.visuals.Ellipse(center=(obstacles.obstacle_point[i, 0], obstacles.obstacle_point[i, 1]),
                                   radius=obstacles.radius[i], color=None, border_color='red')
    view.add(circle)

text = scene.Text(f'agents number: {N}\tcohesion={coeffs[0]}, aligment={coeffs[1]}, separation={coeffs[2]}\n'
                  f'wall avoidance={coeffs[3]}, obstacle avoidance={coeffs[4]}, noise(normal distribution)={coeffs[5]}',
                  color='white', pos=(0.7, 1.4), font_size=10)
fps_on_screen = scene.Text('', pos=(0.15, 1.3), color='white', font_size=10)

view.add(text)
view.add(fps_on_screen)

def update(event):
    """
    Функция для обновления сцены
    """
    boids_first.flocking(coeffs, better_walls_w, obstacles)
    boids_first.propagate(dt)
    arrows.set_data(arrows=boids_first.directions(boids_first.boids, dt))
    fps_on_screen.text = f'fps: {canvas.fps}'
    canvas.update()

if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
