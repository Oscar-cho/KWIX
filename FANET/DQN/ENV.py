import numpy as np
import matplotlib.pyplot as plt
import noise
from tqdm import tqdm

class FANET_ENV:
    def __init__(self, size, reference_distance):
        self.size = size
        self.reference_distance = reference_distance
        
        self.default_reward = -1
        self.terrain_conflict_reward = -100
        self.terminal_reward = 100
        self.hazard_value = 120.0
        self.under_terrain = 125.0
        
        self.height_map = self.height_map_setting() # height_map[y][x] 로 저장됨
        self.base_station = (self.max_elevation_pos[0], self.max_elevation_pos[1], int(self.max_elevation_value)+1)
        self.start_point = (self.min_elevation_pos[0], self.min_elevation_pos[1], int(self.min_elevation))
        #print(f'기지국 위치 {self.base_station}, 시작위치 {self.start_point}')
        self.map_point = {
            'Hazard 1' : {'start': (0, 0, 40), 'end': (10, 10, 60)},
            'Hazard 2' : {'start': (40, 80, 30), 'end': (60, 90, 60)}
        }
        self.loss_map = self.loss_map_setting()     # loss_map[x, y, z]로 결정
    
    def reset(self):
        self.state = self.start_point
        self.counts = 0
        return self.state
    
    def step(self, action):
        self.counts += 1
        x, y, z = self.state
        
        # 초기 3회는 무조건 (0, 0, 3) 으로 이동할 것
        if self.counts <= 3:
            z += 1
        else:
            if action == 0: # +x 방향
                x = min(x + 1, self.size[0] - 1)
            elif action == 1: # -x 방향
                x = max(x - 1, 1)
            elif action == 2: # +y 방향
                y = min(y + 1, self.size[1] - 1)
            elif action == 3: # -y 방향
                y = max(y - 1, 1)
            elif action == 4: # +z 방향
                z = min(z + 1, self.size[2] - 1)
            elif action == 5: # -z 방향
                z = max(z - 1, 1)
        
        reward, stop = self.get_reward_and_terminal(x, y, z)
        self.state = (x, y, z)
        
        return self.state, reward, stop
    
    def get_reward_and_terminal(self, x, y, z):
        if z < self.height_map[y][x] or self.loss_map[x, y, z] == (120.0 or 125.0) or z == 0:  # 지형에 충돌하거나, Hazard Zone에 입장 시 종료 (종료 하지 않음으로 학습함)
            reward = self.terrain_conflict_reward
            stop = False
        elif self.reference_path_loss - 0.1 <= self.loss_map[x, y, z] <= self.reference_path_loss + 0.1:
            reward = self.terminal_reward
            stop = True
        else:
            reward = self.default_reward
            stop = False
        
        if self.counts == 100 * 100:
            stop = True
            
        '''Path Loss 거리에 대한 Terminal 상황 설정할 것'''
        
        return reward, stop
    
    def height_map_setting(self):
        width, height = self.size[0], self.size[1] # x, y축을 기준으로 높이값을 결정
        
        scale = 50 # 노이즈 스케일 (클 수록 더 많은 세부사항)
        octaves = 3 # 옥타브 수 (더 복잡한 노이즈 형태)
        persistence = 0.5 # 감쇠 비율 (0~1 사이 값)
        
        max_elevation = 35 # 최대 높이
        height_scale = 2.5 # 높이 스케일
        
        terrain_data = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                noise_value = noise.pnoise2(y/scale, x/scale, octaves=octaves, persistence=persistence, base = 13)
                terrain_data[y][x] = noise_value * max_elevation * height_scale
                
        terrain_data = np.clip(terrain_data, 0, None)
        
        # 가장 높은 고도를 가진 위치 찾기
        max_elevation_idx = np.argmax(terrain_data)
        max_elevation_idx = np.unravel_index(max_elevation_idx, terrain_data.shape)
        self.max_elevation_pos = max_elevation_idx[1], max_elevation_idx[0]
        self.max_elevation_value = terrain_data[max_elevation_idx]
        
        self.min_elevation, self.min_elevation_pos = self.find_lowest_neighbor(terrain_data, self.max_elevation_pos)
        
        #print(f"가장 높은 고도 위치: {self.max_elevation_pos}, 높이: {self.max_elevation_value}")
        #print(f"가장 낮은 고도 위치: {self.min_elevation_pos}, 높이: {self.min_elevation}")
        
        return terrain_data
    
    def find_lowest_neighbor(self, terrain_data, pos, search_radius=50):
        y, x = pos
        height, width = terrain_data.shape
        min_elevation = float('inf')
        min_elevation_pos = None
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    elevation = terrain_data[ny, nx]
                    if elevation < min_elevation:
                        min_elevation = elevation
                        min_elevation_pos = (nx, ny)
                        
        return min_elevation, min_elevation_pos
    
    def loss_map_setting(self):
        print('Environment Generating...')
        grid = np.zeros((self.size[0], self.size[1], self.size[2]), dtype=np.float32)
        
        for z in tqdm(range(self.size[2])):
            for y in range(self.size[1]):
                for x in range(self.size[0]):
                    if z < self.height_map[y][x]: # Plot 성능에 맞게 수정할 것
                        grid[x, y, z] = self.under_terrain
                    if (x, y, z) == self.base_station:
                        grid[x, y, z] = 0.0
                    else:
                        grid[x, y, z] = self.FSPL(x, y, z)
        
        # Hazard Zone Mapping
        for zone in self.map_point.values():
            start = zone['start']
            end = zone['end']
            grid[start[0]:end[0]+1, start[1]:end[1]+1, start[2]:end[2]+1] = self.hazard_value
                    
        print('Done!')
        return np.round(grid, 1)
    
    def FSPL(self, x, y, z):
        frequency = 2400000000
        bx, by, bz = self.base_station
        
        distance = np.sqrt(abs(x - bx) ** 2 + abs(y - by) ** 2 + abs(z - bz) ** 2)
        
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
        self.reference_path_loss = 20 * np.log10(self.reference_distance) + 20 * np.log10(frequency) - 147.55
        self.reference_path_loss = np.round(self.reference_path_loss, 1)
        
        return path_loss_db
    
class Plot_func:
    def __init__(self, env=FANET_ENV):
        self.env = env
        self.elevation_data = self.env.height_map
        
    def terrain_plot_3d(self):
        x_range = np.linspace(0, self.env.size[0] - 1, self.env.size[0])
        y_range = np.linspace(0, self.env.size[1] - 1, self.env.size[1])
        x, y = np.meshgrid(x_range, y_range)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(x, y, self.elevation_data, cmap='terrain')
        ax.scatter(*self.env.base_station, c='black', marker='s', label='Base Station')
        ax.scatter(*self.env.start_point, c='green', marker='s', label='Start Point')
        hazard_indicies = np.where(self.env.loss_map[:, :, :] == self.env.hazard_value)
        ax.scatter(*hazard_indicies, c='r', marker=',', label='Hazard Zone')
        
        ax.set_zlim(0, 100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain Map')
        ax.legend()
        
        plt.show()
    
    def terrain_plot_2d(self):
        height, width = self.env.height_map.shape
        fliped_data = np.flipud(self.elevation_data)
        
        plt.scatter(*self.env.base_station[:2], c='black', marker='s', label='Base Station')
        plt.scatter(*self.env.start_point[:2], c='red', marker='s', label='Start Point')
        
        plt.imshow(fliped_data, cmap='terrain', extent=[0, width, 0, height])
        plt.colorbar(label='Elevation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Terrain Map')
        plt.legend()
        plt.show()
    
    def signal_map(self, level):
        plt.scatter(*self.env.base_station[:2], c='black', marker='s', label='Base Station')
        plt.scatter(*self.env.start_point[:2], c='red', marker='s', label='Start Point')
        
        plt.imshow(self.env.loss_map[:, :, level].T, origin='lower')
        plt.title("Signal Loss Map")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Path Loss (dB)')
        plt.show()
    
    def path_plot_3d(self, path):
        x_range = np.linspace(0, self.env.size[0] - 1, self.env.size[0])
        y_range = np.linspace(0, self.env.size[1] - 1, self.env.size[1])
        x, y = np.meshgrid(x_range, y_range)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        hazard_indicies = np.where(self.env.loss_map[:, :, :] == self.env.hazard_value)
        ax.scatter(*hazard_indicies, c='r', marker='s', label='Hazard Zone')
        
        ax.plot_surface(x, y, self.elevation_data, cmap='terrain')
        #ax.scatter(*self.env.base_station, c='black', marker='s', label='Base Station')
        ax.scatter(*self.env.start_point, c='green', marker='s', label='Start Point')
        
        positions, collisions = zip(*path)
        x, y, z = np.array(positions).T
        
        for i, collision in enumerate(collisions):
            marker = '.'
            color = 'orange' if collision else 'blue'
            ax.scatter(x[i], y[i], z[i], c=color, marker=marker)
        
        #ax.plot(x, y, z, '.', label='Path')
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_zlim(0, self.env.size[2])
        ax.legend()
        plt.show()
    
    def path_plot_2d(self, path, level):
        height, width = self.env.height_map.shape
        fliped_data = np.flipud(self.elevation_data)
        
        plt.scatter(*self.env.base_station[:2], c='black', marker='s', label='Base Station')
        plt.scatter(*self.env.start_point[:2], c='green', marker='s', label='Start Point')
        
        plt.imshow(fliped_data, cmap='terrain', extent=[0, width, 0, height])
        plt.colorbar(label='Elevation')
        
        positions, collisions = zip(*path)
        x, y, _ = np.array(positions).T
        
        for i, collision in enumerate(collisions):
            marker = '.'
            color = 'red' if collision else 'orange'
            plt.scatter(x[i], y[i], c=color, marker=marker)
        
        plt.title("2D Terrain Path Map")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
        
        '''fig = plt.figure()
        fig.add_subplot(111)
        
        hazard_indicies = np.where(self.env.loss_map[:, :, level] == self.env.hazard_value)
        plt.scatter(*hazard_indicies, c='r', marker='s', label='Hazard Zone')
        
        x, y, _ = np.array(path).T
        plt.plot(x, y, '.', label='Path')
        plt.xlim(0, self.env.size[0])
        plt.ylim(0, self.env.size[1])
        plt.grid(True)
        plt.legend()
        plt.show()'''
        
    def step_count_plot(self, step_count):
        plt.figure()
        plt.plot(step_count)
        plt.xlabel('Episodes')
        plt.ylabel('Step counts')
        plt.title('Step Counts per episodes')
        plt.show()
    
    def reward_plot(self, reward):
        plt.figure()
        plt.plot(reward)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Total Rewards per Epsidoes')
        plt.show()

if __name__ == '__main__':
    env = FANET_ENV((100, 100, 100), 95)
    elevation_data = env.height_map
    
    def plot_3d():
        x_range = np.linspace(0, env.size[0] - 1, env.size[0])
        y_range = np.linspace(0, env.size[1] - 1, env.size[1])
        x, y = np.meshgrid(x_range, y_range)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(x, y, elevation_data, cmap='terrain')
        ax.scatter(*env.base_station, c='black', marker='s', label='Base Station')
        ax.scatter(*env.start_point, c='r', marker='s', label='Start Point')
        
        ax.set_zlim(0, 100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain Map')
        ax.legend()
        
        plt.show()

    def plot_2d():
        height, width = env.height_map.shape
        fliped_data = np.flipud(elevation_data)
        
        plt.scatter(*env.base_station[:2], c='black', marker='s', label='Base Station')
        plt.scatter(*env.start_point[:2], c='red', marker='s', label='Start Point')
        
        plt.imshow(fliped_data, cmap='terrain', extent=[0, width, 0, height])
        plt.colorbar(label='Elevation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Terrain Map')
        plt.legend()
        plt.show()
        
    def signal_map(level):
        plt.scatter(*env.base_station[:2], c='black', marker='s', label='Base Station')
        plt.scatter(*env.start_point[:2], c='red', marker='s', label='Start Point')
        
        plt.imshow(env.loss_map[:, :, level].T, origin='lower')
        plt.title("Signal Loss Map")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Path Loss (dB)')
        plt.show()
    
    #print(f'기지국 위치 확인: {env.height_map[21][39]}')    # [y][x]
    #print(f'signal map 확인: {env.loss_map[39, 21, 39]}')   # [x, y, z] 0이 나와야함 (기지국 위치 값 = 0)
    print('설정한 기준 거리에 대한 Path_loss : ', env.reference_path_loss, 'dB')
    plot_3d()
    plot_2d()
    signal_map(level=3)
    