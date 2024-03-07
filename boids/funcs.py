import numpy as np
from numba import njit, prange


class obstacle:
    """
    Класс преппятствий. Инициализирует препятвия, реализует отталкивание/избегание препятсвий
    """
    def __init__(self, n: int, asp: float):
        """
        Инициализируем препятствия в виде окружностей. Случайным образом
        генерируются их радиусы и центры.

        Параметры
        -------------------------------
        n - число препятсвий
        asp - отношение ширины поля к длине (по умолчанию тогда рассматриваем поле asp x 1)
        """
        self.obstacle_number = n
        self.obstacle_point = np.empty((n, 2))
        self.radius = np.empty(n)
        rng = np.random.default_rng()
        self.obstacle_point[:, 0] = rng.uniform(0., asp, size=n)
        self.obstacle_point[:, 1] = rng.uniform(0., 1., size=n)
        self.radius[:] = rng.uniform(1/30, 1/8, size=n)

    def obstacle_avoidance(self, idx, boids: np.ndarray) -> np.ndarray | float:
        """
        Метод моделирующая избегание агентами препятсвий. Действует следующим образом:
        Для каждого агента, находящегося рядом с препятвием (был выбран коеффициент 20% радиуса)
        строятся два направления: влево и вправо так, чтобы угол между левым направлением и
        вектором соединяющим положение агента и центр препятствия был равен равен углу
        между правым направлением и вектором агент-препятствие. Затем исходя из того к какому
        из направлений ближе агент, вычисляется его скорость и ускорение в сторону ближайщего
        направления.

        Параметры
        -------------------------------
        idx - индекс агента для, которого ищется препятствие в заданной окрестности
        boids - массив с координатми, скоростями и ускорениями агентов

        Возвращаемое значение
        -------------------------------
        Ускорение полученное в результате встречи препятствия
        """
        pos = boids[idx, :2]
        for i, obs in enumerate(self.obstacle_point):
            dist = np.linalg.norm(pos - obs)
            vec = (obs - boids[idx, :2]) / dist
            velocity = (obs - boids[idx, 2:4]) / np.linalg.norm(obs - boids[idx, 2:4])
            if dist < self.radius[i] * 1.2 and dist > self.radius[i]:
                alpha = np.arctan(2 * self.radius[i] / dist)
                mat_left = np.array([[np.cos(1.5*alpha), -np.sin(1.5*alpha)], [np.sin(1.5*alpha), np.cos(1.5*alpha)]])
                mat_right = np.array([[np.cos(1.5*alpha), np.sin(1.5*alpha)], [-np.sin(1.5*alpha), np.cos(1.5*alpha)]])
                vec_left = np.dot(vec, mat_left)
                vec_right = np.dot(vec, mat_right)
                if np.arccos(np.dot(vec_left, velocity)) < np.arccos(np.dot(vec_right, velocity)):
                    boids[idx, 2:4] = 1.5 * vec_left
                    return 1 / 2 * vec_left
                else:
                    boids[idx, 2:4] = 2 * vec_right
                    return - 1 / 2 * vec_right
            if dist < self.radius[i]:
                boids[idx, 2:4] = 1
                return np.ones(2, dtype=np.float_)
        return np.zeros(2, dtype=np.float_)



class boids_agent:
    """
    Класс агентов
    """
    def __init__(self, n: int, asp: float, vrange: tuple[float, float],
                 arange: tuple[float, float], perception: float):
        """
        Инициализируем агентов. Случайным образом генерируются координаты агентов
        на заданном поле.

        Параметры
        -------------------------------
        n - число агентов
        asp - отношение ширины поля к длине (по умолчанию тогда рассматриваем поле asp x 1)
        vrange - ограничения на скорость агентов
        arange - ограничения на ускорение агентов
        perception - граничное расстояние (окрестность, в которой агент считается соседом)
        """
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

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def directions(boids, dt: float) -> np.ndarray:
        """
        Метод для изменения направления движения агентов.

        Параметры
        -------------------------------
        boids - массив с координатми, скоростями и ускорениями агентов
        dt - промежуток времени

        Возвращаемое значение
        -------------------------------
        Массив с измененными координами агентов
        """
        return np.hstack((
            boids[:, :2] - dt * boids[:, 2:4],
            boids[:, :2]
        ))

    @staticmethod
    @njit(fastmath=True)
    def vclip(v: np.ndarray, vrange: tuple[float, float]):
        """
        Метод для контродем скорости (ускорения) агентов.

        Параметры
        -------------------------------
        v - скорость какого-то агента
        vrange - ограничения скорости
        """
        norm = np.zeros(len(v))
        for i in range(len(v)):
            norm[i] = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2)
        mask = norm > vrange[1]
        if np.any(mask):
            v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)


    def propagate(self, dt: float):
        """
        Метод для изменения скорости и положения агентов.
        Осуществляет движение.

        Параметры
        -------------------------------
        dt - промежуток времени
        """
        self.vclip(self.boids[:, 4:6], self.arange)
        self.boids[:, 2:4] += dt * self.boids[:, 4:6]
        self.vclip(self.boids[:, 2:4], self.vrange)
        self.boids[:, 0:2] += dt * self.boids[:, 2:4]

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def distances(boids: np.ndarray) -> np.ndarray:
        """
        Метод для подсчета расстояния между всеми агентами.
        Написанная для компиляции с помощью numba.

        Параметры
        -------------------------------
        boids - массив с координатми агентов

        Возвращаемое значение
        -------------------------------
        Массив с подсчетами расстояниями между всем агентами
        """
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
        """
        Метод для "сплаченности" агентов. Стремление агентов быть рядом,
        двигаться к геометрическому центру.

        Параметры
        -------------------------------
        idx - индекс рассматриваемого агента
        neigh_mask - агенты, которые находятся рядом с рассматриваемым

        Возвращаемое значение
        -------------------------------
        Ускорение направленную к геометрическому центру всех соседей
        """
        center = self.boids[neigh_mask, :2].mean(axis=0)
        a = (center - self.boids[idx, :2]) / self.perception
        return a



    def separation(self,
                   idx: int,
                   neigh_mask: np.ndarray) -> np.ndarray:
        """
        Метод для "отдаления" агентов. Стремление агентов двигаться по отдельности, а не рядом

        Параметры
        -------------------------------
        idx - индекс рассматриваемого агента
        neigh_mask - агенты, которые находятся рядом с рассматриваемым

        Возвращаемое значение
        -------------------------------
        Ускорение направленное в строну противоположную от движения других агентов
        """
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
        """
        Метод для "выравнивания" агентов. Стремление агентов двигаться в одном направлении.

        Параметры
        -------------------------------
        idx - индекс рассматриваемого агента
        vrange - ограничения cкорости

        Возвращаемое значение
        -------------------------------
        Ускорение направленное в строну движения соседей
        """
        v_mean = self.boids[neigh_mask, 2:4].mean(axis=0)
        a = (v_mean - self.boids[idx, 2:4]) / (2 * self.vrange[1])
        return a

    @staticmethod
    @njit(fastmath=True)
    def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
        """
        Метод для гладкого скачка. Позволяет длеать ускорение окло стен более гладким,
        то есть нет резких скачков, оно меняется плавно.

        Параметры
        -------------------------------
        edge0 - первая граница области
        edge1 - вторая граница области
        x - аргумент для которого строится гладкий скачок

        Возвращаемое значение
        -------------------------------
        Массив из значений полинома третьей степени с гладким скачком
        """
        x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
        return x * x * (3.0 - 2.0 * x)

    def better_walls(self, param: float):
        """
        Метод, дающий агентам рассположенным около стен ускорение, применяет
        функцию smoothstep.

        Параметры
        -------------------------------
        param - показатель для гладкого скачка

        Возвращаемое значение
        -------------------------------
        Массив из ускорений для того, что агенты не могли выплыть далеко
        за пределы границы
        """
        x = self.boids[:, 0]
        y = self.boids[:, 1]
        w = param

        a_left = self.smoothstep(self.asp * w, 0.0, x)
        a_right = -self.smoothstep(self.asp * (1.0 - w), self.asp, x)

        a_bottom = self.smoothstep(w, 0.0, y)
        a_top = -self.smoothstep(1.0 - w, 1.0, y)

        return np.column_stack((a_left + a_right, a_bottom + a_top))

    def flocking(self, coeffs, order, obs: obstacle):
        """
        Метод позволяющий проходиться по всем агентам и изменять скорость, ускорение,
        направление движения в зависимости от положения агентов в текущей момент времени.

        Параметры
        -------------------------------
        coeffs - коеффициенты симуляции, вносящие вклад в ускорение (cohesion, alignment,
        order - порядок smothstep
        obs - класс препятсвий, которые встречаются на пути
        """
        D = self.distances(self.boids)
        N = self.boids.shape[0]
        D[range(N), range(N)] = self.perception + 1
        mask = D < self.perception
        wal = self.better_walls(order)
        for i in range(N):
            if not np.any(mask[i]):
                coh = np.zeros(2)
                alg = np.zeros(2)
                sep = np.zeros(2)
            else:
                coh = self.cohesion(i, mask[i])
                alg = self.alignment(i, mask[i])
                sep = self.separation(i, mask[i])
            obs_avoid = obs.obstacle_avoidance(i, self.boids)
            a = coeffs[0] * coh + coeffs[1] * alg + \
                coeffs[2] * sep + coeffs[3] * wal[i] + \
                coeffs[4] * obs_avoid + coeffs[5] * np.random.normal(0,1,2)

            self.boids[i, 4:6] = a