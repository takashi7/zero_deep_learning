from and_gate2 import AND
from nand_or_gate import NAND, OR

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    for x in [(0,0),(1,0),(0,1),(1,1)]:
        y = XOR(x[0], x[1])
        print(str(x) + "->" + str(y))
