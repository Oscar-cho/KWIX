# DQN을 이용하여 자율주행 자동차 생성

# 환경 생성

# Importing the libraries
import numpy as np
from random import random, randint
import time

# Kivy packet import
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from deep_q_learning.py
from deep_q_learning import Dqn

# 오른쪽 클릭으로 빨간색 점을 추가하지 않기 위해 이 줄 추가
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# 지도에 모래를 그릴 때 마지막 점을 memory에 남기기 위해 사용되는 last_x와 last_y를 도입
last_x = 0
last_y = 0
n_points = 0
length = 0

# AI의 brain, 행동 반경 및 reward 변수 만들기
brain = Dqn(4,3,0.9)            # 4개의 input, 3개의 action , gamma는 0.9로 설정
action2rotation = [0,20,-20]    # 회전 각도 20도에서 -20도 사이로 결정
reward = 0

# 맵 시작
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))  # 폭과 길이 설정
    goal_x = 20                          # x 목표 위치
    goal_y = largeur - 20                # y 목표 위치
    first_update = False

# last distance 초기화
last_distance = 0

# Car class 정의

class Car(Widget):
    
    angle = NumericProperty(0)                            # 지도의 x축과 자동차의 축 사이의 각도
    rotation = NumericProperty(0)                         # 자동차가 마지막으로 회전한 방향 -> 행동 실행시 자동차 회전하는걸 보기 위함
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)  # sensor 1 위치
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)  # sensor 2 위치
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)  # sensor 3 위치
    signal1 = NumericProperty(0)                           # sensor 1으로 받은 신호
    signal2 = NumericProperty(0)                           # sensor 2로 받은 신호
    signal3 = NumericProperty(0)                           # sensor 3으로 받은 신호

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos      # pos:자동차 위치
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# game class 정의

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global reward
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width  # 길이(세로)
        largeur = self.height  # 높이
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        state = [orientation, self.car.signal1, self.car.signal2, self.car.signal3]  # sensor들이 보내는 정보는 signal 1~3으로 알려준다.
        action = brain.update(state, reward)
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            reward = -0.2
            if distance < last_distance:
                reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            reward = -1
        if self.car.y < 10:
            self.car.y = 10
            reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            reward = -1                             # 자동차가 지도를 벗어나지 않도록 끝에 도달하면 reward = -1 부여

        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y

        last_distance = distance

# pointing tool 생성

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_yd
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# API Buttons 생성 (clear, save and load 버튼)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# 해당 내용 running 하기 위해
if __name__ == '__main__':
    CarApp().run()
