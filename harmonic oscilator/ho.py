import numpy as np
import random
import matplotlib.pyplot as plt
import sys

global dt          # segment time
global maxtime     # end point time
global replicas    # replicas of particles
global ER          # reference energy
global E_record    # record E during running
global xmin, xmax  # range of coordinate
global Nmax        # maximum number for particles
global N_record    # record N during running
global N0, N1      # old and new number of particles
global nb          # number of boxes to accumulate wave function
global wave_func
global coordinate

maxtime = 100
dt = 0.1
ER=0.0
nb = 200
N0 = 500
xmin = -20.0
xmax = 20.0
Nmax = 2000
N_record = []
E_record = []

def potential(x):
	return 0.5*x**2

def weight(x):
	global ER
	w = np.exp(dt*(0.5*x**2-ER))
	return w

def initialize(x0=0):
	global replicas, dt, Nmax 
	global N_record, N0 
	global E_record, ER
	replicas=np.zeros([Nmax,2])
	# row indice is the index of particles
	# first column is the state of particles (active or inactive)
	# second colum is the coordinates of particles in 1-d space
	
	# activate N0 particles for initialize 
	N_record.append(N0)
	for i in range(N0):
		replicas[i][0] = 1
	E_record.append(ER)

def diffuse():
	global dt, xmin, xmax, replicas
	sigma = np.sqrt(dt)
	for i in range(Nmax):
		if replicas[i][0] == 1:
			g = np.random.normal(loc = 0.0, scale = 1.0, size = None)
			replicas[i][1] += g
			if replicas[i][1] < xmin: replicas[i][1] = xmin
			if replicas[i][1] > xmax: replicas[i][1] = xmax

def birth_death():
	global replicas, N1, N_record
	# traverse all active particles
	for i in range(Nmax):
		if replicas[i][0] == 1:
			# get their weights
			w = weight(replicas[i][1])
			m = np.min([int(w+random.random()), 3])
			# birth or death
			if m == 0:
				replicas[i][0] = 0
			elif m == 1:
				pass
			else:
				for k in range(m):
					for kk in range(Nmax):
						if replicas[kk][0] == 0:
							replicas[kk][0] = 1
							replicas[kk][1] = replicas[i][1]
	# count the number of particles after birth-death
	N1 = 0
	for i in range(Nmax):
		if replicas[i][0] == 1:
			N1 += 1
	N_record.append(N1)
	

def tune_Er():
	global ER, E_record, N1, N0, dt
	ER = ER + (1-1.0*N1/N0)/dt
	E_record.append(ER)
	N0 = N1

def is_in_box(x, box):
	'''
	box is a list with 2 entities
	'''
	if box[0]<=x<box[1]:
		return True
	else:
		return False
	
def get_wavefunc(self):
	global xmin, xmax, nb, Nmax
	global replicas
	global wave_func
	global coordinate
	x = np.zeros(nb)
	w = np.zeros(nb)

	# get all boxes
	tmp = np.linspace(xmin, xmax, nb+1)
	boxes=[]
	
	for i in range(nb):
		boxes.append([tmp[k],tmp[k+1]])
	# accumulate
	k = 0
	for box in self.boxes:
		x[k] = 0.5*(box[0]+box[1])
		for p in range(Nmax):
			if replicas[p][0] == 1:
				if is_in_box(replicas[p][1], box):
					w[k] += 1
		k += 1

	A = np.sqrt(np.sum(1.0*w**2))
	psi = np.sqrt(1.0*w/A)
	wave_func = psi
	coordinate = x

		
def flushPrint(string,num):
	sys.stdout.write('\r')
	sys.stdout.write(string+'%s' % num)
	sys.stdout.flush()


def main():
	global dt, maxtime
	global E_record, N_record
	global wave_func, coordinate
	initialize()
	t = np.arange(0, maxtime, dt)
	for tt in t:
		flushPrint("Current t: ", tt)
		diffuse()
		birth_death()
		tune_Er()

	get_wavefunc()
	E_ave = []
	for i in range(len(E_record)):
		E_ave.append[np.mean(E_record[:i+1])]

	plt.figure(figsize=[16,9])
	plt.subplot(131)
	plt.title('Wave function')
	plt.plot(coordinate, wave_func, 'b-')

	plt.subplot(132)
	plt.title('Energy')
	plt.xlabel('time')
	plt.ylabel('<E_R>')
	plt.plot(t, E_ave)

	plt.subplot(133)
	plt.plot('Number of Particles')
	plt.xlabel('time')
	plt.ylabel('N')
	plt.plot(t, N_record, 'b-')

	plt.show()

if __name__ == "__main__":
	main()









