import copy
import random
import os
import pickle
from files import *
from PIL import Image


def GetData(img):
    pixels = list(img.getdata())
    rez = [list(pixels[i]) for i in range(len(pixels))]
    return rez


def SetData(img, pixels, h, w):
    a_len = len(pixels)
    for i in range(a_len):
        pixels[i] = tuple(pixels[i])
    mult = (h * w)
    for i in range(mult):
        a = (i % w, i // w)
        img.putpixel(a, pixels[i])


def CreateMatrx(p, n, m):
    mult = n * m * 3
    rez = [[random.randint(-1, 1)/100
            for j in range(p)]
           for i in range(mult)]
    return rez


def matrixTranspose(M): # stack overflow
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]


def multipl(A, B): # stack overflow
    C = [[0 for row in range(len(B[0]))] for col in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                C[i][j] += A[i][k] * B[k][j]
    return C


def colorConvert(color):
    rez = 2*color/255
    return rez - 1


def rev_colorConvert(color):
    a = (color + 1)/2
    return a*255

##############
def paramsForRefData(h, n, w,m,i):
    arr_params = []
    par_1 = (w//m)
    arr_func = [i//par_1, (h//n)*par_1, i % par_1] # row L col
    arr_params.extend(arr_func)
    for k in range(1):
        arr_func.clear()
        if (arr_func == []):
            pass
    return arr_params


def DataRefRange(Arr, j, w, k,X):
    for a in range(3):
        value = colorConvert(Arr[j*w + k][a])
        X.append([value])
    return X


def FullRangeRefData(params, n, m, Arr, w, X):
    for j in range(params[0] * n, params[2] * n + n):
        for k in range(params[2] * m, params[2] * m + m):
            rez = DataRefRange(Arr, j, w, k,X)
    return rez



def refData(Arr, n, m, h, w, refer_v):
    i = 0
    params = paramsForRefData(h, n, w,m,i)
    for i in range(params[1]):
        X = []
        rez = FullRangeRefData(params, n, m, Arr, w, X)
        refer_v.append(rez)
    return refer_v


######################
def infoForRef(h, w):
    return [[0 for row in range(3)]
            for col in range(int(h*w))], -1


def rangeForRef(A, x, y, info, a, w, b):
    for z in range(3):
        rev_col = int(rev_colorConvert(A[x][y][0]))
        info[a*w + b][z], y = rev_col, y + 1


def dataRefer(A, n, m, h, w):
    info, x = infoForRef(h, w)
    for i in range(0, h // m):
        for j in range(0, w // n):
            y, x = 0, x + 1
            for a in range(i * m, i * m + m):
                for b in range(j * n, j * n + n):
                    rangeForRef(A, x, y, info, a, w, b)
    return info


##################

def delta(m1, m2):
    return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]


def errCount(delta_x):
    len_d = len(delta_x[0])
    result = 0
    for i in range(len_d):
        rez = delta_x[0][i] * delta_x[0][i]
        result += rez
    return result


def alphaM(matrix, alpha): # stackoverflow
    return [[alpha * matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]


def errSum(E):
    result = []
    for i in range(len(E)):
        result.append(E[i])
    return sum(result)

####################################


def createW1_W2(p,n,m):
    W1 = CreateMatrx(p, n, m)
    W2 = matrixTranspose(W1)
    a = [W1,W2]
    return a


def trainRange(rang, img, alpha, W1, W2, E):
    for i in range(int(rang)):
            Y = (multipl(matrixTranspose(img[i]), W1))
            buff = copy.deepcopy(W2)
            W2 = delta(W2, alphaM(multipl(matrixTranspose(Y), delta(multipl(Y, W2), matrixTranspose(img[i]))), alpha))
            W1 = delta(W1, alphaM(multipl(matrixTranspose(matrixTranspose(img[i])), multipl(delta(multipl(Y, W2), matrixTranspose(img[i])), matrixTranspose(buff))), alpha))
            E.append(errCount(delta(multipl(Y, W2), matrixTranspose(img[i]))))
    a = [E, W1, W2]
    return a


def Train(image, n, m, p, height, width, e, alpha, count, E):
    create = createW1_W2(p, n, m)
    W1, W2 = create[0], create[1]
    img = refData(GetData(image), n, m, height, width, [])
    while errSum(E) > e:
        E = []
        count += 1
        rang = height / n * width / m
        OurRange = trainRange(rang, img, alpha, W1, W2, E)
        E = OurRange[0]
        W1 = OurRange[1]
        W2 = OurRange[2]
        print("Step "
              + str(count) +
              '\n')
        print(errSum(E))
    setDataFile(n, m, p, W1, W2)

#####################################


def paramsComp(file):
    width, height = Image.open(file).size
    n, m, p, W1, W2 = getDataFile()
    img = refData(GetData(Image.open(file)), n, m, height, width, [])
    a = [height, n,width, m,img, W1, W2]
    return a


def comressRange(form, params, comp):
    for i in range(int(form)):
        X = matrixTranspose(params[4][i])
        comp.append((multipl(X, params[5])))
    return comp


def compress(file, compImg):
    params = paramsComp(file)
    height, n, width, m = params[0], params[1], params[2], params[3]
    form = height / n * width / m
    compImg = comressRange(form, params, compImg)
    with open('compress/compress_img.bin', "wb") as file:
        pickle.dump([compImg, width, height], file)

##########################################


def openDecompress(file, rez,width, height, Y):
    with open("compress/" + file, "rb") as file:
        info = pickle.load(file)
        Y = info[rez - 10]
        width = info[rez - 9]
        height = info[rez - 8]
    a = [Y, width, height]
    return a


def rezultDec(form,new_img,width,Y, W2,height,n,m):
    for i in range(int(form)):
        new_img.append(matrixTranspose(multipl(Y[i], W2)))
    result = Image.new("RGB", (width, height), "#FF0000")
    par_1 = dataRefer(new_img, n, m, height, width)
    SetData(result, par_1, height, width)
    return result


def decompress(file, rez, width, height, Y):
    params = openDecompress(file, rez,width, height, Y)
    Y, width, height = params[0], params[1], params[2]
    new_img = []
    n, m, p, W1, W2 = getDataFile()
    form = height / n * width / m
    show = rezultDec(form,new_img,width,Y,W2,height,n,m)
    show.show()


