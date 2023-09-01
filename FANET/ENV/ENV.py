import numpy as np
import matplotlib.pyplot as plt
import noise
from tqdm import tqdm

class FANET_ENV:
    def __init__(self, size, target_dB, target_level, max_steps):
        self.size = size
        self.target_dB = target_dB
        self.target_level = target_level
        self.max_steps = max_steps
        
        self.map_point = {
            'Hazard 1' : {'start': (0, 0, 20), 'end': (10, 10, 25)}
        }
        
        self.terrain_map = self.terrain_map_setting()
        self.base_station = (self.max_elevation_pos[0], self.max_elevation_pos[1], int(self.max_elevation_value)+1)
        self.start_point = (self.min_elevation_pos[0], self.min_elevation_pos[1], int(self.min_elevation))
        
        self.loss_map = self.loss_map_setting()
        self.target_pos = self.find_coordinate()
    
    def reset(self):
        self.state = self.start_point
        self.counts = 0
        return self.state
    
    def empty_step(self, action):
        self.counts += 1
        x, y, z = self.state
        
        if self.counts <= 2:
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
        
        if self.counts == self.max_steps:
            stop = True
        else:
            stop = False
        
        if self.target_pos[0] - 5 <= x <= self.target_pos[0] + 5 and self.target_pos[1] - 5 <= y <= self.target_pos[1] + 5 and self.target_pos[2] - 3 <= z <= self.target_pos[2] + 3:
            reward = 10000 
            stop = True
        else:
            reward =  -1
        
        return self.state, reward, stop
    
    def step(self, action):
        self.counts += 1
        x, y, z = self.state
        
        if self.counts <= 2:
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
        if self.counts == self.max_steps:
            stop = True
        else:
            stop = False
        
        if z <= self.terrain_map[y][x] or self.loss_map[x, y, z] == (120.0 or 121.0) or z == 0:
            return -500, stop
        elif self.target_pos[0] - 5 <= x <= self.target_pos[0] + 5 and self.target_pos[1] - 5 <= y <= self.target_pos[1] + 5 and self.target_pos[2] - 3 <= z <= self.target_pos[2] + 3:
            return 10000, True
        else:
            return -self.reward_mapping(x, y, z), stop
        
        '''elif self.reward_mapping(x, y, z) >= 100:
            return -10, stop
        elif self.reward_mapping(x, y, z) >= 90:
            return -9, stop
        elif self.reward_mapping(x, y, z) >= 80:
            return -8, stop
        elif self.reward_mapping(x, y, z) >= 70:
            return -7, stop
        elif self.reward_mapping(x, y, z) >= 60:
            return -6, stop
        elif self.reward_mapping(x, y, z) >= 50:
            return -5, stop
        elif self.reward_mapping(x, y, z) >= 40:
            return -4, stop
        elif self.reward_mapping(x, y, z) >= 30:
            return -3, stop
        elif self.reward_mapping(x, y, z) >= 20:
            return -2, stop
        else: #self.reward_mapping(x, y, z) >= 10
            return -1, stop'''
        
        
            
    def terrain_map_setting(self):
        print('Environment Generating...')
        width, height = self.size[0], self.size[1] # x, y축을 기준으로 높이값을 결정
        
        scale = 50 # 노이즈 스케일 (클 수록 더 많은 세부사항)
        octaves = 3 # 옥타브 수 (더 복잡한 노이즈 형태)
        persistence = 0.5 # 감쇠 비율 (0~1 사이 값)
        
        max_elevation = 35 # 최대 높이
        height_scale = 1.5 # 높이 스케일
        
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
        grid = np.zeros((self.size[0], self.size[1], self.size[2]), dtype=np.float32)
        
        for z in tqdm(range(self.size[2])):
            for y in range(self.size[1]):
                for x in range(self.size[0]):
                    if z < self.terrain_map[y][x]:
                        grid[x, y, z] = 121.0
                    elif (x, y, z) == self.base_station:
                        grid[x, y, z] = 0.0
                    else:
                        grid[x, y, z] = self.FSPL(x, y, z)
        
        # Hazard Zone Mapping
        for zone in self.map_point.values():
            start = zone['start']
            end = zone['end']
            grid[start[0]:end[0]+1, start[1]:end[1]+1, start[2]:end[2]+1] = 120.0

        return np.round(grid, 1)
    
    def FSPL(self, x, y, z):
        frequency = 2400000000
        bx, by, bz = self.base_station
        
        distance = np.sqrt(abs(x - bx) ** 2 + abs(y - by) ** 2 + abs(z - bz) ** 2)
        
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
        
        return round(path_loss_db, 2)
    
    def find_coordinate(self):
        coordinates = []
        for z in tqdm(range(self.size[2])):
            for y in range(self.size[1]):
                for x in range(self.size[0]):
                    if (x, y, z) == self.base_station:
                        continue
                    current_dB = self.FSPL(x, y, z)
                    if np.isclose(current_dB, self.target_dB):
                        if z > self.terrain_map[y][x] and self.loss_map[x, y, z] != (120 or 121):
                            if z == self.target_level:
                                coordinates.append((x,y,z))
        print('Done !')
        return coordinates[-1]
    
    def reward_mapping(self, x, y, z):
        dx, dy, dz = self.target_pos
        distance = np.sqrt(abs(x - dx) ** 2 + abs(y - dy) ** 2 + abs(z - dz) ** 2)
        
        return round(distance)