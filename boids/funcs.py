import numpy as np
from numba import njit, prange
from numba.experimental import jitclass

spec = [
    ('boids', float64[:]),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass
class boids_agent:
    def __init__(self, n: int, asp: float, vrange: tuple[float, float],
                 arange: tuple[float, float], perception: float):
        self.boids = np.zeros((n, 6), dtype=np.float64)
        self.vrange = vrange
        self.arange = arange
        self.perception = perception
        self.asp = asp
        rng = np.random.default_rng()
        self.boids[:, 0] = rng.uniform(0., asp, size=n)
        self.boids[:, 1] = rng.uniform(0., 1., size=n)
        alpha = rng.uniform(0, 2 * np.pi, size=n)
        v = rng.uniform(*vrange, size=n)
        c, s = np.cos(alpha), np.sin(alpha)
        self.boids[:, 2] = v * c
        self.boids[:, 3] = v * s
        pass

    def directions(self, dt: float) -> np.ndarray:
        """
        :param boids:
        :param dt:
        :return: array N x (x0, y0, x1, y1) for arrow painting
        """
        return np.hstack((
            self.boids[:, :2] - dt * self.boids[:, 2:4],
            self.boids[:, :2]
        ))

    def vclip(self, v: np.ndarray, vrange: tuple[float, float]):
        norm = np.linalg.norm(v, axis=1)
        mask = norm > vrange[1]
        if np.any(mask):
            v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)

    def propagate(self, dt: float,):
        self.vclip(self.boids[:, 4:6], self.arange)
        self.boids[:, 2:4] += dt * self.boids[:, 4:6]
        self.vclip(self.boids[:, 2:4], self.vrange)
        self.boids[:, 0:2] += dt * self.boids[:, 2:4]

    @staticmethod
    # @njit(parallel=True, fastmath=True)
    def distances(boids: np.ndarray) -> np.ndarray:
        n, m = boids.shape
        D = np.empty((n, n), dtype=float)
        for i in prange(n):
            for j in range(n):
                delta = boids[i] - boids[j]
                D[i, j] = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
        return D

    def cohesion(self,
                 idx: int,
                 neigh_mask: np.ndarray) -> np.ndarray:
        center = self.boids[neigh_mask, :2].mean(axis=0)
        a = (center - self.boids[idx, :2]) / self.perception
        return a

    def separation(self,
                   idx: int,
                   neigh_mask: np.ndarray) -> np.ndarray:
        neighbs = self.boids[neigh_mask, :2] - self.boids[idx, :2]
        norm = np.linalg.norm(neighbs, axis=1)
        mask = norm > 0
        if np.any(mask):
            neighbs[mask] /= norm[mask].reshape(-1, 1)
        d = neighbs.mean(axis=0)
        norm_d = np.linalg.norm(d)
        if norm_d > 0:
            d /= norm_d
        return -d

    def alignment(self,
                  idx: int,
                  neigh_mask: np.ndarray) -> np.ndarray:
        v_mean = self.boids[neigh_mask, 2:4].mean(axis=0)
        a = (v_mean - self.boids[idx, 2:4]) / (2 * self.vrange[1])
        return a


    def smoothstep(self, edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
        x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
        return x * x * (3.0 - 2.0 * x)

    def better_walls(self, param: float):
        x = self.boids[:, 0]
        y = self.boids[:, 1]
        w = param

        a_left = self.smoothstep(self.asp * w, 0.0, x)
        a_right = -self.smoothstep(self.asp * (1.0 - w), self.asp, x)

        a_bottom = self.smoothstep(w, 0.0, y)
        a_top = -self.smoothstep(1.0 - w, 1.0, y)

        return np.column_stack((a_left + a_right, a_bottom + a_top))


    def flocking(self,
                 coeffs: np.ndarray,
                 order: float):
        D = self.distances(self.boids)
        N = self.boids.shape[0]
        D[range(N), range(N)] = self.perception + 1
        mask = D < self.perception
        wal = self.better_walls(order)
        for i in prange(N):
            if not np.any(mask[i]):
                coh = np.zeros(2)
                alg = np.zeros(2)
                sep = np.zeros(2)
            else:
                coh = self.cohesion(i, mask[i])
                alg = self.alignment(i, mask[i])
                sep = self.separation(i, mask[i])
            a = coeffs[0] * coh + coeffs[1] * alg + \
                coeffs[2] * sep + coeffs[3] * wal[i]
            self.boids[i, 4:6] = a


class obstacle:
    def __init__(self, n: int, asp: float, sepsration_speed: float):
        self.obstacle_number = n
        self.obstacle_point = np.empty((n, 2))
        self.sepsration_speed = sepsration_speed
        rng = np.random.default_rng()
        self.obstacle_point[:, 0] = rng.uniform(0., asp, size=n)
        self.obstacle_point[:, 1] = rng.uniform(0., 1., size=n)
        self.radius = np.array([asp/40] * n) + 0.01

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def distance_obstacle_agent(boids: np.ndarray, obs: np.ndarray) -> np.ndarray:
        n, m = boids.shape
        k, l = obs.shape
        D = np.empty((n, k), dtype=float)
        for i in prange(n):
            for j in range(k):
                delta = boids[i] - obs[j]
                D[i, j] = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
        return D

    # считаем расстояние от каждого бойда до центра круга
    # если оно около припятвия, то разворачиваем бойда (добавляем separation)
    #
    def separation_from_obstacle(self, boids_cur: boids_agent) -> np.ndarray:
        D = self.distance_obstacle_agent(boids_cur.boids, self.obstacle_point)
        for i in range(self.obstacle_number):
            mask = D[:, i] < self.radius[i]
            boids_cur.boids[mask, 2:] *= -1

            # neighbs /= norm[mask].reshape(-1, 1)
            # d = neighbs.mean(axis=0)
            # norm_d = np.linalg.norm(d)
            # if norm_d > 0:
            #     d /= norm_d
            # return -d

    def separation(self, boids_cur: boids_agent):

        D = self.distance_obstacle_agent(boids_cur.boids, self.obstacle_point)
        for i in range(self.obstacle_number):
            mask = D[:, i] < 1.05 * self.radius[i]

            boids_cur.boids[mask, 2:4] = -1 * (self.radius[i] - boids_cur.boids[mask, :2]) / 100
            #
            # self.velocityX += -1 * (obstacle.real_x - self.rect.x) / self.obstacle_avoidance_weight
            # self.velocityY += -1 * (obstacle.real_y - self.rect.y) / self.obstacle_avoidance_weight




