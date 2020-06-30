
from math import *
from datasets import *
def paint_pics(data,name):
    fig=plt.figure()
    l=len(data)
    up=ceil(sqrt(l))
    down = floor(sqrt(l))
    print('up',up,'down',down)
    for i in range(l):
        plt.subplot(up,down,i+1)
        plt.imshow(data[i].permute(1,2,0).squeeze())
    plt.savefig(name+'.png')

def paint_pics_1(name,*data ):
    fig = plt.figure()
    for d in data:
        l = len(d)
        up = ceil(sqrt(l))
        down = floor(sqrt(l))
        print('up', up, 'down', down)
        for i in range(l):
            plt.subplot(up, down, i + 1)
            plt.imshow(d[i].squeeze())
    plt.savefig(name + '.png')
    # plt.show()

def paint_scatters(name: object, *args: object) -> object:
    fig=plt.figure()
    # print(len(args))
    # clr={'r','b'}
    for i,data in enumerate(args):
        data_x=[d[0] for d in data]
        data_y=[d[1] for d in data]
        plt.scatter(data_x,data_y)
        # print('i',i)
    # plt.scatter(data1[0],data1[1],c='green')
    plt.savefig(name+'.png')
    # plt.show()
