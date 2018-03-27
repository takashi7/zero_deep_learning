import numpy as np 

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for x in [(0,0), (1,0),(0,1),(1,1)]:
        y_NAND = NAND(x[0],x[1])
        y_OR = OR(x[0],x[1])
        print(str(x) + "->" + str(y_NAND)+ "->" + str(y_OR))
