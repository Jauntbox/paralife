########################################################################
# life.py
#
# Author: Kevin Moore
#
# Date started: 6/2/15
# 
# Description: A parallel implementation of Conway's Game of Life in Python.
########################################################################

from sys import exit
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI

class Life:
	'''inputs:
			N: length & width of the simulation in number of cells'''
	def __init__(self, N):
		self.N = N
		self.grid = np.random.choice(2, (N,N))
		self.aug_grid = self.form_aug_grid(self.grid)
		self.fig = plt.figure()

		#self.fig, self.ax = plt.subplots()
		#self.mat = self.ax.matshow(self.grid)

	def __str__(self):
		return str(self.grid)

	'''Forms an augmented grid so that calculations can be done entirely in place:'''
	def form_aug_grid(self,a):
		return np.lib.pad(a, ((1,1),(1,1)), 'wrap')

	'''Function that sums all the neighbors of the (row,col)'th entry of the grid'''
	def neighbor_sum(self,row,col):
		#aug_grid[row+1][col+1] is the element we're interested in in the augmented grid.
		#It's guaranteed to have all 8 neighbors defined by constructrion.
		return self.aug_grid[row+2][col+2] + self.aug_grid[row+2][col+1] + self.aug_grid[row+2][col] + \
			self.aug_grid[row+1][col+2] + self.aug_grid[row+1][col] + \
			self.aug_grid[row][col+2] + self.aug_grid[row][col+1] + self.aug_grid[row][col]

	'''Function that updates the grid according to the rules of the game of life using
		a toroidal geometry (wrapping in both directions - like asteroids)'''
	def update(self,i):
		for row,col in np.ndindex(self.grid.shape):
			if self.neighbor_sum(row,col) == 3:
				self.grid[row][col] = 1
			elif self.neighbor_sum(row,col) != 2:
				self.grid[row][col] = 0
		self.aug_grid = self.form_aug_grid(self.grid)
		
		plt.cla()	#Clear the plot so things draw much faster
		return plt.imshow(self.grid, interpolation='nearest')

	def update_para(self,i):
		#print 'Updating!'
		newgrid = self.grid
		for row in range(N):
			if(row % mpi_size == rank):
				#print 'proc ',rank,' working on row ',row
				for col in range(N):
					if self.neighbor_sum(row,col) == 3:
						self.grid[row][col] = 1
					elif self.neighbor_sum(row,col) != 2:
						self.grid[row][col] = 0
					#Broadcast just this finished row to the rest of the procs:

		#comm.barrier()
		for row in range(N):
			#print rank, self.grid.shape, self.grid[row].shape
			self.grid[row] = comm.bcast(self.grid[row], root=row % mpi_size)

		#print newgrid
		#self.grid = newgrid
		if(rank == 0):
			self.aug_grid = self.form_aug_grid(self.grid)
		self.aug_grid = comm.bcast(self.aug_grid,root=0)


		plt.cla()	#Clear the plot so things draw much faster
		return plt.imshow(self.grid, interpolation='nearest')

	'''Display one step of the simulation'''
	def show(self):
		#fig, ax = plt.subplots()
		#mat = ax.matshow(self.grid)
		plt.imshow(self.grid, interpolation='nearest')
		#im = ax.imshow(self.grid, interpolation='nearest')
		plt.show()

	'''Play a movie of the simulation'''
	def movie(self):
		#fig, ax = plt.subplots()
		#mat = ax.matshow(self.grid)
		anim = animation.FuncAnimation(self.fig, self.update, interval=10)
		plt.show()

	'''Play a movie of the simulation with parallelization'''
	def movie_para(self):
		#fig, ax = plt.subplots()
		#mat = ax.matshow(self.grid)
		anim = animation.FuncAnimation(self.fig, self.update_para, interval=10)
		plt.show()

N = 500 #Size of grid for the game of life
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
#Figure out MPI division of tasks (how many rows per processor)
rows_per_proc = N / mpi_size
if(rows_per_proc != int(N*1.0 / mpi_size)):
	if(comm.Get_rank() == 0):
		print "Matrix size not evenly divisible by number of processors. Use \
			different values."
		exit()
elif(mpi_size > 1):
	if(comm.Get_rank() == 0):
		print "Parallelizing with ",mpi_size," processors."
else:
	print "Running in serial mode."

game = Life(N) #Initialize the game with a grid of size N
#Make proc 0 have the true matrix - send it to the others:
game.grid = comm.bcast(game.grid, root=0)
game.aug_grid = game.form_aug_grid(game.grid)
rank = comm.Get_rank()
print rank, game
print ""

if(mpi_size == 1):
	game.movie()
else:
	num_steps = 100
	#game.movie_para()
	for i in range(num_steps):
		game.update_para(i)
		#if(rank == 0):
		#	plt.cla()
		#	game.show()


#Now all processors have the same game grid, let's see if we can split up the 
#updating accordingly:
#newgrid = game.grid
#for row in range(N):
#	if((rank+1) % (row+1) == 0):
#		for col in range(N):
#			if game.neighbor_sum(row,col) == 3:
#				game.grid[row][col] = 1
#			elif game.neighbor_sum(row,col) != 2:
#				game.grid[row][col] = 0
			#Broadcast just this finished row to the rest of the procs:
#		newgrid[row] = comm.bcast(game.grid[row],root=rank)
#comm.barrier()
#game.grid = newgrid

#num=3
#game.update_para(num)
#print rank, game
#print ""

#comm.barrier()
#game.update_para(num)
#print rank, game
#print ""

#if(rank == 0):
	#game.movie_para()
	#fig = plt.figure()
	#anim = animation.FuncAnimation(game.fig, game.update_para, interval=10)
	#plt.show()

#print rank, game
#print ""


#This section will make it work with multiple processors enabled using mpiexec
#if(rank == 0):
#	fig = plt.figure()
#	anim = animation.FuncAnimation(fig, game.update, interval=10)
#	plt.show()
#else:
#	while(True):
#		sleep(1.0)


#rank = game.comm.Get_rank()
#if(rank == 0):
#	game.movie()

#Want parallel version to look like:
	#Setup data for all processors (do one random and send to others?)
	#Split work for the update task between processors
		#Communicate the work back to other processors comm.bcast()?
	#Animate data (maybe with just one processor?)


#game.show()
#game.update()
#game.show()
#game.update()
#game.show()

#fig, ax = plt.subplots()
#mat = ax.matshow(game.grid)
#anim = animation.FuncAnimation(fig, game.update, frames=30, interval=1, blit=True)
#plt.show()

#Initialize the grid:
#grid = np.random.choice(2, (N,N))
#Define an augmented grid so that calculations can be done entirely in place:
#aug_grid = np.lib.pad(grid, ((1,1),(1,1)), 'wrap')
#aug_grid = form_aug_grid(grid)
#print grid
#print aug_grid

#This iterates through the array with the column index changing the fastest:
#for row,col in np.ndindex(grid.shape):
#	print grid[row][col], aug_grid[row+1][col+1], neighbor_sum(row,col)

#This iterates through the array using two explicit for loops:
#for row in range(N):
#	for col in range(N):
#		print game.grid[row][col]

#Main code
#comm = MPI.COMM_WORLD
#size = MPI.COMM_WORLD.Get_size()
#rank = comm.Get_rank()