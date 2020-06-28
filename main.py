from experiment import *
from models.res18 import resnet18

if __name__ == "__main__":
    model = resnet18()
    experiment(model)
    experiment(model,2,20)