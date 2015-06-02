########################################################################
# life.py
#
# Author: Kevin Moore
#
# Date started: 6/2/15
# 
# Description: A parallel implementation of Conway's Game of Life in Python.
########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI

class Life:
	def __init__(self, N):
		self.grid = np.random.choice(2, (N,N))
		self.aug_grid = self.form_aug_grid(self.grid)
		self.fig = plt.figure()
		#self.fig, self.ax = plt.subplots()
		#self.mat = self.ax.matshow(self.grid)

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
		
		plt.cla()	#Clear the plot so things draw 
		return plt.imshow(self.grid, interpolation='nearest')

	def show(self):
		fig, ax = plt.subplots()
		mat = ax.matshow(self.grid)
		#im = ax.imshow(self.grid, interpolation='nearest')
		plt.show()

	def movie(self):
		#fig, ax = plt.subplots()
		#mat = ax.matshow(self.grid)
		anim = animation.FuncAnimation(self.fig, self.update, interval=10)
		plt.show()

N = 40 #Size of grid for the game of life
game = Life(N) #Initialize the game with a grid of size N
game.movie()

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


#Main code
comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = comm.Get_rank()