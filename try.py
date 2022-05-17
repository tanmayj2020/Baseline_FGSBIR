import pickle
import os
import numpy as np
from bresenham import bresenham

coordinate_path = os.path.join('Dataset', "ShoeV2" , "ShoeV2" + '_Coordinate')
root_dir = os.path.join('Dataset',"ShoeV2")
with open(coordinate_path, 'rb') as fp:
    Coordinate = pickle.load(fp)

Train_Sketch = [x for x in Coordinate if 'train' in x]
Test_Sketch = [x for x in Coordinate if 'test' in x]
#print(Train_Sketch)
#print(Test_Sketch)

#print("_".join(Train_Sketch[0].split('/')[-1].split('_')[:-1]))
a = np.array(Coordinate[Train_Sketch[1]])
b = np.array(Coordinate[Train_Sketch[123]])

def mydrawPNG(vector_image):
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    final_list = []
    for i in range( 0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        final_list.extend([list(j) for j in cordList])
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
    return final_list

a = mydrawPNG(a)
b = mydrawPNG(b)
import random
random.shuffle(a)
random.shuffle(b)
a = a[:300]
b = b[:300]


import math
from ortools.linear_solver import pywraplp
import numpy as np
import random
from bresenham import bresenham


def euclidean_distance(x , y):
    return math.sqrt(sum((a-b)**2 for (a ,b) in zip(x ,y)))

def data_matrix(p1 , p2):
    data_ = []
    for x in p1:
        x_list = []
        for y in p2:
            x_list.append(euclidean_distance(x , y))
        data_.append(x_list)
    return data_



def getAssignment(p1 , p2):
    assert(len(p1) == len(p2)) , "p1 and p2 must have same number of points"
    num_points = len(p1)
    data_ = data_matrix(p1 , p2)
    solver= pywraplp.Solver.CreateSolver("SCIP")

    # Creatinng the optimization variables
    x = {}
    for i in range(num_points):
        for j in range(num_points):
            x[i , j] = solver.IntVar(0 , 1 , '')
    
    # Adding the bijective Constraints
    for i in range(num_points):
        solver.Add(solver.Sum([x[i,j] for j in range(num_points)]) == 1)
    
    for j in range(num_points):
        solver.Add(solver.Sum([x[i,j] for i in range(num_points)]) == 1)
    
    # Objective function 
    objective_terms =[]
    for i in range(num_points):
        for j in range(num_points):
            objective_terms.append(data_[i][j] * x[i , j])
    solver.Minimize(solver.Sum(objective_terms))

    status = solver.Solve()
    ans_dict = {}
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print("Bijection success")
        print("EMD (minimum work) = " , solver.Objective().Value()/num_points , "\n")
        for i in range(num_points):
            for j in range(num_points):
                if x[i , j].solution_value() > 0.5:
                    ans_dict[i] = j
        return ans_dict
    else:
        print('No solution found.')



def interpolation(p1 , p2 , lmbda , ans_dict):
    assert 0 <= lmbda <= 1 , "lambda out of bounds"
    n = len(p1)
    if lmbda == 0.0:
        return p2
    if lmbda == 1.0:
        return p1
    intermediate_point_list = []
    for i in range(n):
        j = ans_dict[i]
        point1 = np.array(p1[i])
        point2 = np.array(p2[j])
        
        intermediate_point = (lmbda) * point1 + (1-lmbda) * point2
        intermediate_point_list.append(intermediate_point)
    intermediate_point_list = np.array(intermediate_point_list)
    return intermediate_point_list

import matplotlib.pyplot as plt

def visualize_numpy_points(a , b , ans_dict):
    for lmbda in [0.0 , 0.1 , 0.2, 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ,1.0]:
        interpolation_ = interpolation(a , b , lmbda , ans_dict)
        data_x = [coordinate[0] for coordinate in interpolation_]
        data_y = [coordinate[1] for coordinate in interpolation_]
        axs = plt.axes()
        axs.set_facecolor("black")
        plt.scatter(data_x , data_y , color="white" , s=10)
        plt.title(str(lmbda))
        plt.show()


#ans_dict = getAssignment(a , b)
#visualize_numpy_points(a , b , ans_dict)

import numpy as np
import scipy.ndimage as nd


def preprocess(sketch_points, side=256.0):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_Sketch(sketch_points):
    sketch_points = np.array(sketch_points)
    p1 = preprocess(sketch_points , 256.0)
    raster_image = np.zeros((int(256), int(256)), dtype=np.float32)
    for coordinate in p1:
        if (coordinate[0] > 0 and coordinate[1] > 0) and (coordinate[0] < 256 and coordinate[1] < 256):
                raster_image[int(coordinate[1]), int(coordinate[0])] = 255.0
    raster_image = nd.binary_dilation(raster_image) * 255.0
    
    return raster_image

a = rasterize_Sketch(a)
from PIL import Image
a =Image.fromarray(a).convert("RGB")
a.show()
