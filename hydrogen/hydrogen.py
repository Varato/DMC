import numpy as np
from numpy import linalg as la
import random
import matplotlib.pyplot as plt
import sys
from pprint import pprint
import copy

global ceil, floor
floor = -20
ceil = 20

class particle:
	def __init__(self, r):
		'''
		r1, r2: the initial positions of 2 electrons
		'''
		self.old_r = r
		self.r = r
		# self.m = 1
	def diffuse(self, sigma):
		global floor, ceil
		self.old_r = copy.deepcopy(self.r)
		while 1:
			tmp = copy.deepcopy(self.r)
			g = np.random.normal(loc = 0.0, scale = 1.0, size = 3)
			tmp += sigma*g
			er = 0.001
			jdg = (la.norm(tmp) > er)
			if jdg:
				self.r = tmp
				break
		for d in range(3):
			if self.r[d] < floor: self.r[d] = floor
			if self.r[d] > ceil: self.r[d] = ceil

	def get_m(self, weight):
		self.m = min([int(weight+random.random()), 3])

class molecule:
	def __init__(self, x0 = [0.,0.,1.], N0 = 500, Nmax = 2000, dt = 0.1):
		'''
		x0: all particles initialize at here
		N0: number of initial particles
		'''
		self.Nmax = Nmax
		self.N0 = self.N = N0
		self.N_new = 0
		self.dt = dt
		self.sigma = np.sqrt(dt)
		self.replicas = []
		self.E_record = []
		self.N_record = []

		for i in range(self.N):
			self.replicas.append( particle(np.array(x0)) )

		self.Er = 0.0


	def potential(self, r):
		global R
		V = -1.0/la.norm(r)
		return V

	def getV_bar(self):
		V=[]
		for parti in self.replicas:
			V.append(self.potential(parti.r))
		V_bar=np.mean(V)
		return V_bar

	def weight(self, r, old_r):
		w = np.exp(-self.dt*(self.potential(r)+self.potential(old_r)-2.0*self.Er)/2)
		return w

	def clear(self):
		tmp = []
		for parti in self.replicas:
			if parti.m == 0:
				pass
			else:
				for i in range(parti.m):
					# if len(tmp) > self.Nmax: break
					tmp.append(particle(parti.r))
		self.replicas = copy.deepcopy(tmp) 

	def one_Monto_Carlo_step(self):
		for parti in self.replicas:
			parti.diffuse(self.sigma)

	def birth_death(self):
		for parti in self.replicas:
			w = self.weight(parti.r, parti.old_r)
			parti.get_m(w)
		self.clear()
		self.N_new = len(self.replicas)
		self.N_record.append(self.N_new)

	def tune_Er(self):
		# if self.E_record:
		# 	E_best = min(self.E_record)
		# else:
		# 	E_best = self.Er
		self.Er = self.getV_bar() + (1.0 - 1.0*self.N/self.N0 )/self.dt
		self.E_record.append(self.Er)
		self.N=copy.deepcopy(self.N_new)

	def getElectron_Cloud(self):
		coordinates = []
		for parti in self.replicas:
			coordinates.append(parti.r)
		coordinates = np.array(coordinates)
		return coordinates


def flushPrint(num1, num2, num3):
	sys.stdout.write('\r')
	sys.stdout.write("CURRENT t: %.2f; N: %d; E_R: %.2f" % (num1, num2, num3))
	sys.stdout.flush()


def main(N0=500, dt=0.1, tmax=100):
	mo = molecule(N0=N0, dt=dt)
	t = np.arange(0, tmax, mo.dt)
	print "============= DMC start ============="
	for tt in t:
		flushPrint(tt+dt, mo.N, mo.Er)
		mo.one_Monto_Carlo_step()
		mo.birth_death()
		mo.tune_Er()
	print
 
	E_ave = []
	for i in range(len(mo.E_record)):
		E_ave.append(np.mean(mo.E_record[:i+1]))
	error = np.abs(np.array(E_ave)+0.5)
	coordinates = mo.getElectron_Cloud()
	np.save("data.npy", coordinates)

	plt.figure(figsize=[14,11])

	plt.subplot(211)
	plt.plot(t, E_ave, 'r-', label='$<E_R>$')
	plt.plot([0,100],[-0.5, -0.5], 'k--')
	# plt.plot(t, error, 'b-', label='$Error$')
	plt.xlabel('$t$')
	plt.ylabel('$<E_R>$')
	plt.legend(loc="upper right")


	plt.subplot(212)
	plt.plot(t, mo.N_record, 'b-')
	plt.plot([0,100],[N0, N0],'r-',linewidth=2)
	plt.xlabel('$t$')
	plt.ylabel('$N$')

	plt.savefig("hydrogen.png")	
	plt.show()


if __name__=='__main__':
	main(N0 = 1000)










