import numpy as np
import matplotlib.pyplot as mpl

print("sdfsdf")
dat = np.genfromtxt('housing.txt')
#x=dat[1:3][3]
#x1=dat[:,4]
#print(x1)
#x2=x1[0:250]
#x3=dat[:,14]
#x4=x1[0:250]

x=dat[0:250,13]

print(x)
x=[0,1,2,3,4,5,6,7,8,9,10,11,12]
y=[67.2,0,0,56.4,50.9,33.7,0,52.4,48.5,0,62.1,0,0]
mpl.plot(x,y)
mpl.show()
