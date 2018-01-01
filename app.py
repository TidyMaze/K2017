import tkinter
import tkinter.messagebox
import math
import random
import time
import cProfile
import threading
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

ai = len(sys.argv) > 1 and sys.argv[1] == "AI"
human = not ai

POD_RADIUS = 10
STEP = 5

SPEED = 4

NB_SENSORS = 9
ROTATE_ANGLE = 5

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 1000

CELL_WIDTH = 30
CELL_HEIGHT = 30

X = []
y = []

class State:
    def __init__(self):
        self.pod = None

    def __str__(self):
        return 'state [ pod : ' + str(self.pod) + ' ]'

class Pod:
    """ un acteur d'un joueur """
    def __init__(self):
        self.position = None
        self.speed = Vector(0,0)
        self.angle = 0
        self.power = 0
        self.circle = None
        self.line = None
        self.sensorsDst = []
        self.sensorsDraw = []

    def updateDraw(self,canvas,grid):
        # print("update draw")

        currentDraw = canvas.coords(self.circle)
        # currentDraw = canvas.coords(self.line)
        dX = self.position.x - (currentDraw[0]+POD_RADIUS)
        dY = self.position.y - (currentDraw[1]+POD_RADIUS)
        # print(self.position.x,self.position.y)
        if(dX == 0 and dY == 0): return
        canvas.move(self.circle, dX, dY)
        # canvas.move(self.line, dX, dY)

        collisions = findSensorsCollisions(self,grid)

        # print("collisions : ", collisions)

        def updateCollision(i,c):
            # print(self.sensorsDraw)
            currentDraw = canvas.coords(self.sensorsDraw[i])
            (dst,point) = c
            dX = point.x - (currentDraw[0]+POD_RADIUS)
            dY = point.y - (currentDraw[1]+POD_RADIUS)
            if(dX == 0 and dY == 0): return
            canvas.move(self.sensorsDraw[i], dX, dY)

        for i, item in enumerate(collisions):
            updateCollision(i, item)

    def reset(self,grid):
        startGridPoint = [Point(j,i) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] == 'S'][0]
        # print(startGridPoint)

        startPoint = gridPointToPoint(startGridPoint.x,startGridPoint.y)
        # print(startPoint)

        self.position = Point(startPoint.x + CELL_WIDTH/2, startPoint.y + CELL_HEIGHT/2)
        self.angle = 0

        speedX = math.cos(math.radians(self.angle))*SPEED
        speedY = math.sin(math.radians(self.angle))*SPEED

        self.speed = Vector(speedX,speedY)

    def __str__(self):
        return 'position : ' + str(self.position) + ', speed : ' + str(self.speed) + ', angle : ' + str(self.angle) + ', power : ' + str(self.power)

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return '('+str("%.2f" % self.x)+','+str("%.2f" % self.y)+')'
    def add(self, obj2D):
        return Vector(self.x + obj2D.x, self.y + obj2D.y)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return '('+str("%.0f" % self.x)+','+str("%.0f" % self.y)+')'
    def apply(self, vector):
        return Point(self.x + vector.x, self.y + vector.y)

class KeyManager:
    def __init__(self, widget):
        self.states = {}
        widget.bind("<KeyPress>", self.keydown)
        widget.bind("<KeyRelease>", self.keyup)

    def keyup(self,e):
        self.setState(e.keycode,False)
        # self.printStates()
    def keydown(self,e):
        self.setState(e.keycode,True)
        # self.printStates()
    def printStates(self):
        print(self.states)
    def setState(self,keyCode, state):
        self.states[keyCode]=state
    def getState(self,keyCode):
        return self.states[keyCode] if keyCode in self.states else False

def createCircle(canvas, x, y, radius, color):
    return canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline=color, tag='moving')

def createPod(canvas, pod, color, grid):
    collisions = findSensorsCollisions(pod, grid)
    pod.sensorsDraw = list(map(lambda c: createCircle(canvas, c[1].x, c[1].y, 10, 'red'), collisions))
    pod.circle = createCircle(canvas, pod.position.x, pod.position.y, POD_RADIUS, color)
    # pod.line = canvas.create_line(
    #     pod.position.x,
    #     pod.position.y,
    #     pod.position.x + math.cos(math.radians(pod.angle))*POD_RADIUS,
    #     pod.position.y + math.sin(math.radians(pod.angle))*POD_RADIUS,
    #     fill=color,
    #     tag='moving'
    # )

def drawRectangle(canvas, x, y, width, height, color):
    canvas.create_rectangle(x, y, x+width, y+height, fill=color)

def inRange(val, min, max):
    return val >= min and val <= max

def empty(list):
    return len(list) == 0

def lost(state, grid):
    currentGridPoint = pointToGridPoint(state.pod.position.x, state.pod.position.y)
    return grid[currentGridPoint.y][currentGridPoint.x] == '#'

def win(state, grid):
    currentGridPoint = pointToGridPoint(state.pod.position.x, state.pod.position.y)
    return grid[currentGridPoint.y][currentGridPoint.x] == '9'

def dst(p1, p2):
    return math.sqrt(math.pow(p2.x-p1.x,2)+math.pow(p2.y-p1.y,2))


def calcSensorCollision(pod, grid, a):
    vX = math.cos(math.radians(a)) * STEP
    vY = math.sin(math.radians(a)) * STEP
    i = 0
    while(True):
        curX = pod.position.x + vX * i
        curY = pod.position.y + vY * i
        point = pointToGridPoint(curX,curY)
        gridX = point.x
        gridY = point.y
        if grid[gridY][gridX] == '#':
            dst = i*STEP
            cX = math.cos(math.radians(a)) * dst
            cY = math.sin(math.radians(a)) * dst
            return (dst, Point(pod.position.x + cX, pod.position.y + cY))
        i+=1


def findSensorsCollisions(pod, grid):
    base_angle = (180/(NB_SENSORS-1))
    # print("base_angle", base_angle)
    angles = list(map(lambda i: base_angle*i-90 + pod.angle, range(NB_SENSORS)))
    # print("angles length", len(angles))
    return list(map(lambda a: calcSensorCollision(pod, grid, a),angles))

def train():
    with open('data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile,  delimiter=' ', quotechar='|')
        Xextracted = []
        yextracted = []
        for row in reader:
            print("Cake: ", row)
            Xextracted.append(list(map(int,row[0:9])))
            yextracted.append(list(map(int,row[9:10])))
        print("Xextracted: ", Xextracted)
        print("yextracted: ", yextracted)

    classifier = MLPClassifier(hidden_layer_sizes=(300,300,300),max_iter=1000000)

    # INPUT : X et y

    # Tranformations
    scaler = StandardScaler().fit(Xextracted)
    Xnp = scaler.transform(Xextracted)
    ynp = np.array(yextracted)



    X_train, X_test, y_train, y_test = train_test_split(Xnp, ynp, test_size=0.83)

    print("Dim : ", X_train.shape, y_train.shape)

    # OUTPUT
    classifier.fit(X_train, y_train.ravel())
    score = classifier.score(X_test, y_test)
    y_pred = classifier.predict(Xnp)

    # DISPLAY
    print("Score : ", score)
    return (scaler, classifier)

def writeData():
    global X,y

    print(X, y)

    dataToWrite = [X[i] + y[i] for i in range(len(X))]

    with open('data.csv', 'a+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(dataToWrite)
        X = []
        y = []

def showMove(canvas, state, grid, root, keyManager, classifier=None, scaler=None):
    # print("SHOW_MOVE")



    if human:
        # LEFT
        if keyManager.getState(37): state.pod.angle -= ROTATE_ANGLE
        # RIGHT
        if keyManager.getState(39): state.pod.angle += ROTATE_ANGLE
        # if keyManager.getState(38): state.pod.angle
        # if keyManager.getState(40): state.pod.angle

        if keyManager.getState(37) and not keyManager.getState(39): output = [1]
        elif keyManager.getState(39) and not keyManager.getState(37): output = [2]
        else: output = [3]

        collisions = findSensorsCollisions(state.pod,grid)
        distances = list(map(lambda c: c[0],collisions))

        X.append(distances)
        y.append(output)
        print(distances, output)
    else:
        collisions = findSensorsCollisions(state.pod,grid)
        distances = list(map(lambda c: c[0],collisions))
        X_scaled = scaler.transform(np.array(distances).reshape(1, -1))
        y_pred = classifier.predict(X_scaled)
        print("Pred : ", y_pred)

        if y_pred[0] == 1: state.pod.angle -= ROTATE_ANGLE
        elif y_pred[0] == 2: state.pod.angle += ROTATE_ANGLE
        else:
            print("nothing : ", y_pred[0])

    state = updateGame(state, state.pod.angle, state.pod.power)
    state.pod.updateDraw(canvas,grid)

    if human:
        if lost(state, grid):
            writeData()
            state.pod.reset(grid)
        if win(state, grid):
            writeData()
            state.pod.reset(grid)
    else:
        if lost(state, grid):
            c = train()
            scaler = c[0]
            classifier = c[1]
            state.pod.reset(grid)
    canvas.after(33, lambda : showMove(canvas, state, grid, root, keyManager, classifier, scaler))

def updateGame(state, angle, power):
    # print('updating : before : ' + str(state))
    newState = State()
    newState.pod = Pod()
    newState.pod.circle = state.pod.circle
    newState.pod.line = state.pod.line
    newState.pod.sensorsDraw = state.pod.sensorsDraw
    newState.pod.angle = angle
    newState.pod.power = power

    speedX = math.cos(math.radians(newState.pod.angle))*SPEED
    speedY = math.sin(math.radians(newState.pod.angle))*SPEED

    newState.pod.speed.x = speedX
    newState.pod.speed.y = speedY
    newState.pod.position = state.pod.position.apply(newState.pod.speed)
    # newState.history = list(state.history)
    # newState.history.append((angle,power))
    # print('updating : after : ' + str(newState))
    return newState

def main():
    grid = [
 "########################################",
 "#      #     #   1 #     #     #    2  #",
 "# S       #     #     #     #     #    #",
 "#     #############################    #",
 "# ###########      ###  # #### ###     #",
 "#                    ####         3    #",
 "#                       #    ###########",
 "#                       #  5           #",
 "#              # #      #   ############",
 "#             #   ##    #       6      #",
 "#             # 9  ##   ############## #",
 "#             #     ##               # #",
 "#             #      ################  #",
 "#             #            8          7#",
 "#             #######################  #",
 "#                                    # #",
 "########################################"
]

    # print(grid)
    createWindow(grid)

def gridPointToPoint(xGrid, yGrid):
    return Point(xGrid * CELL_WIDTH, yGrid * CELL_HEIGHT)

def pointToGridPoint(x, y):
    return Point(math.floor(x/CELL_WIDTH), math.floor(y/CELL_HEIGHT))

def show(grid, canvas, root, keyManager, classifier=None, scaler=None):
    pod = Pod()
    pod.reset(grid)

    state = State()
    state.pod = pod

    drawCircuit(canvas, grid)
    createPod(canvas, pod, 'blue', grid)

    showMove(canvas, state, grid, root, keyManager, classifier, scaler)

    root.mainloop()

def drawCircuit(canvas, grid):

    map = {
        "#" : 'black',
        " " : 'white',
        "S" : 'green',
        "1" : 'yellow',
        "2" : 'yellow',
        "3" : 'yellow',
        "4" : 'yellow',
        "5" : 'yellow',
        "6" : 'yellow',
        "7" : 'yellow',
        "8" : 'yellow',
        "9" : 'yellow'
    }

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            drawRectangle(
                canvas,
                j*CELL_WIDTH,
                i*CELL_HEIGHT,
                CELL_WIDTH,
                CELL_HEIGHT,
                map[grid[i][j]]
            )


def createWindow(grid):
    root = tkinter.Tk()
    canvas = tkinter.Canvas(root, bg="black", height=len(grid)*CELL_HEIGHT, width=len(grid[0])*CELL_WIDTH)
    canvas.focus_set()

    canvas.pack()
    km = KeyManager(canvas)

    if not human:
        c = train()
        scaler = c[0]
        classifier = c[1]
        show(grid, canvas, root, km, classifier, scaler)
    else:
        show(grid, canvas, root, km, None, None)
main()
