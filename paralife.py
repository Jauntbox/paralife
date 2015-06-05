########################################################################
# paralife.py
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
	'''N: length & width of the simulation in number of cells'''
	def __init__(self, N):
		self.N = N
		self.comm = MPI.COMM_WORLD
		self.mpi_size = comm.Get_size()
		self.rank = comm.Get_rank()
		self.grid = np.random.choice(2, (N,N))
		self.aug_grid = self.form_aug_grid(self.grid)

		#If we're doing the point-to-point communication method, then each process
		#only needs to hold a small part of the entire simulation. The first and
		#last rows/columns of self.proc_grid will be 'ghost' rows/columns for the 
		#periodic boundaries. The actual rows we care about will range from indices
		#1-N (indices 0 and N+1 are ghost cells). Likewise, the columns we will care
		#about have indices 1-self.cols_per_proc with the first & last being ghosts.
		if(self.mpi_size > 1):
			recv_buf = np.arange(N+2)
			self.cols_per_proc = int(np.ceil(N*1.0/self.mpi_size))
			self.proc_grid = np.random.choice(2, (N+2,self.cols_per_proc+2))

			#Constant initialized grid for testing with N=4 and 2 procs:
			#if(self.rank==0):
			#	self.proc_grid = np.array([[0,1,1,0],[1,0,1,0],[0,1,0,0],[1,0,0,0],[1,1,0,0],[0,1,0,1]])
			#if(self.rank==1):
			#	self.proc_grid = np.array([[1,0,0,1],[0,1,0,1],[1,0,1,1],[0,1,1,1],[0,0,1,1],[1,0,1,0]])

			#Still life for testing with N=4 and 2 procs:
			#if(self.rank==0):
			#	self.proc_grid = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]])
			#if(self.rank==1):
			#	self.proc_grid = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]])

			#Make the bottom/top boundaries reflective:
			self.proc_grid[0,:] = self.proc_grid[N,:]
			self.proc_grid[N+1,:] = self.proc_grid[1,:]

			#Set up all the ghost cells between the processes:
			self.comm_ghosts()

		self.fig = plt.figure()

	'''Communicates the ghost cells between the processes'''
	def comm_ghosts(self):
		if(self.mpi_size > 1):
			recv_buf_ary = np.arange(N+2)	#Buffer for receiving an array
			recv_buf_num = np.arange(1)		#Buffer for receiving a number
			#Integer tags for where data was *sent* from (so as to ensure it arrives in the correct buffer):
			[left_tag, right_tag, corner_tl_tag, corner_bl_tag, corner_tr_tag, corner_br_tag] = range(6)

			#Send ghost cells & columns from the neighboring processes:
			comm.Isend(np.array(self.proc_grid[:,1]), dest=(self.rank-1)%self.mpi_size, tag=left_tag)
			comm.Isend(np.array(self.proc_grid[:,self.cols_per_proc]),dest=(self.rank+1)%self.mpi_size, tag=right_tag)
			#If we're at the edges, then also send the corners:
			#if(self.rank==0):
			#	comm.Isend(self.proc_grid[1,1], dest=self.mpi_size-1, tag=corner_tl_tag)
			#	comm.Isend(self.proc_grid[N,1], dest=self.mpi_size-1, tag=corner_bl_tag)
			#if(self.rank==self.mpi_size-1):
			#	comm.Isend(self.proc_grid[1,self.cols_per_proc], dest=0, tag=corner_tr_tag)
			#	comm.Isend(self.proc_grid[N,self.cols_per_proc], dest=0, tag=corner_br_tag)

			#Receive ghost cells & columns from the neighboring processes:
			comm.Recv(recv_buf_ary, source=(self.rank-1)%self.mpi_size, tag=right_tag)
			self.proc_grid[:,0] = recv_buf_ary
			comm.Recv(recv_buf_ary, source=(self.rank+1)%self.mpi_size, tag=left_tag)
			self.proc_grid[:,self.cols_per_proc+1] = recv_buf_ary
			#if(self.rank==0):
			#	comm.Recv(recv_buf_num, source=self.mpi_size-1, tag=corner_tr_tag)
			#	self.proc_grid[N+1,0] = recv_buf_num
			#	comm.Recv(recv_buf_num, source=self.mpi_size-1, tag=corner_br_tag)
			#	self.proc_grid[0,0] = recv_buf_num
			#if(self.rank==self.mpi_size-1):
			#	comm.Recv(recv_buf_num, source=0, tag=corner_tl_tag)
			#	self.proc_grid[N+1,self.cols_per_proc+1] = recv_buf_num
			#	comm.Recv(recv_buf_num, source=0, tag=corner_bl_tag)
			#	self.proc_grid[0,self.cols_per_proc+1] = recv_buf_num


	def __str__(self):
		return str(self.grid)


	'''Forms an augmented grid so that calculations can be done entirely in place:'''
	def form_aug_grid(self,a):
		return np.lib.pad(a, ((1,1),(1,1)), 'wrap')


	'''Function that sums all the neighbors of the (row,col)'th entry of the grid using
		data from the global simulation'''
	def neighbor_sum(self,row,col):
		#aug_grid[row+1][col+1] is the element we're interested in in the augmented grid.
		#It's guaranteed to have all 8 neighbors defined by constructrion.
		return self.aug_grid[row+2][col+2] + self.aug_grid[row+2][col+1] + self.aug_grid[row+2][col] + \
			self.aug_grid[row+1][col+2] + self.aug_grid[row+1][col] + \
			self.aug_grid[row][col+2] + self.aug_grid[row][col+1] + self.aug_grid[row][col]

	'''Function that sums all the neighbors of the (row,col)'th entry of the grid using
		data local to this process (self.proc_grid)'''
	def neighbor_sum_local(self,row,col):
		return self.proc_grid[row+1][col+1] + self.proc_grid[row+1][col] + self.proc_grid[row+1][col-1] + \
			self.proc_grid[row][col+1] + self.proc_grid[row][col-1] + \
			self.proc_grid[row-1][col+1] + self.proc_grid[row-1][col] + self.proc_grid[row-1][col-1]


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


	'''Updates the matrix by splitting things into columns and then using point to
		point communication to share boundary columns between processes. Each 
		process now gets a contiguous set of columns to operate on, so the number
		of processes should not be greater than the number of columns (otherwise
		several of them will not be used).'''
	def update_para_1d_dec_point_to_point(self,i):
		#Make a temporary grid so that we don't overwrite cells while working on
		#the update step:
		temp_grid = self.proc_grid

		for row in range(1,N+1):
			for col in range(1,self.cols_per_proc+1):
				if self.neighbor_sum_local(row,col) == 3:
					temp_grid[row][col] = 1
				elif self.neighbor_sum_local(row,col) != 2:
					temp_grid[row][col] = 0
		self.proc_grid = temp_grid

		#Now send the updated columns to the necessary
		comm.barrier()
		self.comm_ghosts()


	'''Updates the simulation using a 1d domain decomposition: every process works
		on a group of rows determined by that processor's rank, then the results are
		broadcast to all the other processes so that each process has the entire updated 
		simulation domain after each step'''
	def update_para_1d_dec(self,i):
		#print 'Updating!'
		for row in range(N):
			if(row % mpi_size == rank):
				#print 'proc ',rank,' working on row ',row
				for col in range(N):
					if self.neighbor_sum(row,col) == 3:
						self.grid[row][col] = 1
					elif self.neighbor_sum(row,col) != 2:
						self.grid[row][col] = 0

		#comm.barrier()
		#Broadcast just this finished row to the rest of the procs:
		for row in range(N):
			#print rank, self.grid.shape, self.grid[row].shape
			self.grid[row] = comm.bcast(self.grid[row], root=row % mpi_size)

		#Is it faster for one process to calculate the aug_grid and then broadcast it
		#to the other processes, or for all the processes to calculate it? I assume
		#This varies with process number and problem size, but didn't notice any
		#difference using 2 processes on my laptop.
		if(rank == 0):
			self.aug_grid = self.form_aug_grid(self.grid)
		self.aug_grid = comm.bcast(self.aug_grid,root=0)

		plt.cla()	#Clear the plot so things draw much faster
		return plt.imshow(self.grid, interpolation='nearest')


	'''Updates the simulation using a 2d domain decomposition: each process gets 
		a rectangular (square?) patch to work on '''
	def update_para_2d_dec(self,i):
		#print 'Updating!'
		for row in range(N):
			if(row % mpi_size == rank):
				#print 'proc ',rank,' working on row ',row
				for col in range(N):
					if self.neighbor_sum(row,col) == 3:
						self.grid[row][col] = 1
					elif self.neighbor_sum(row,col) != 2:
						self.grid[row][col] = 0

		#comm.barrier()
		#Broadcast just this finished row to the rest of the procs:
		for row in range(N):
			#print rank, self.grid.shape, self.grid[row].shape
			self.grid[row] = comm.bcast(self.grid[row], root=row % mpi_size)

		#Is it faster for one process to calculate the aug_grid and then broadcast it
		#to the other processes, or for all the processes to calculate it? I assume
		#This varies with process number and problem size, but didn't notice any
		#difference using 2 processes on my laptop.
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


	'''Play a movie of the simulation with parallelization.'''
	def movie_para(self):
		#fig, ax = plt.subplots()
		#mat = ax.matshow(self.grid)
		anim = animation.FuncAnimation(self.fig, self.update_para_1d_dec, interval=10)
		plt.show()

N = 40 #Size of grid for the game of life
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


do_movie = False
game = Life(N) #Initialize the game with a grid of size N

#Make proc 0 have the true matrix - send it to the others:
game.grid = comm.bcast(game.grid, root=0)
game.aug_grid = game.form_aug_grid(game.grid)
rank = comm.Get_rank()
#print rank, game
#print ""
if(game.mpi_size > 1):
	print rank, game.proc_grid

#Running with the movie enabled in parallel will cause an animation to pop up
#for each process. This is pretty impractical and only intended for testing
#purposes. I don't know if there's a way to make only one process display the
#movie, but nothing I've tried works so far.
if(do_movie):
	if(mpi_size == 1):
		game.movie()
	else:
		game.movie_para()
#For running without the movie, just run the simulation for num_steps.
else:
	num_steps = 100
	for i in range(num_steps):
		game.update_para_1d_dec_point_to_point(i)
		print rank, game.proc_grid
		#game.update_para_1d_dec(i)


#Want parallel version to look like:
	#Setup data for all processors (do one random and send to others?)
	#Split work for the update task between processors
		#Communicate the work back to other processors comm.bcast()?
	#Animate data (maybe with just one processor?)

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