# USAGE
# python seam_carving.py --image image.jpg --file file.txt

# castle.jpg taken from Wikipedia article on seam carving

# import the necessary packages
from numpy import linalg as LA
import pickle
import numpy as np
import argparse
import cv2
import copy

# set values is used for the forward computation of edge energies. It also points to the pixel with the least cumulative energy among all pixels' upward neighbors (excluding the first row)
def set_values(mat,energy,disp):
	for x in range(1,len(mat)):
		for y in range(len(mat[0])):
			if x==0:
				disp[x][y] = 0 #garbage value
			else:
				#setting values , looking up to see which is the best option
				if y==0:
					if energy[x-1][y] <= energy[x-1][y+1]:
						disp[x][y] = 0
						mat[x][y] =mat[x-1][y]
					else:
						disp[x][y] = 1
						mat[x][y] += mat[x-1][y+1] 
				elif y==len(mat[0])-1:
					if energy[x-1][y] <= energy[x-1][y-1]:
						disp[x][y] = 0
						mat[x][y] += mat[x-1][y]
					else:
						disp[x][y] = -1
						mat[x][y] += mat[x-1][y-1]
				else:
					if energy[x-1][y]<= energy[x-1][y-1] and energy[x-1][y]<= energy[x-1][y+1]:
						disp[x][y] = 0							
						mat[x][y] += mat[x-1][y]
					elif energy[x-1][y-1] <= energy[x-1][y+1]:
						disp[x][y] = -1
						mat[x][y] += mat[x-1][y-1]
					else:
						disp[x][y] = 1
						mat[x][y] += mat[x-1][y+1]
	return

#for calculating energy for one pixel in an image
def get_energy(img,x,y,norm=2):
	e=0	
	#getting energy along both directions
	if x==0:
		e+= 2 * np.linalg.norm(np.abs(np.array(img[0][y][:]) - np.array(img[1][y][:])),norm)
	elif x==len(img)-1:
		e+= 2 * np.linalg.norm(np.abs(np.array(img[x][y][:]) - np.array(img[x-1][y][:])),norm)
	else:
		e+= np.linalg.norm(np.abs(np.array(img[x][y][:]) - np.array(img[x-1][y][:])),norm) + np.linalg.norm(np.abs(np.array(img[x][y][:]) - np.array(img[x+1][y][:])),norm)

	if y==0:
		e+= 2 * np.linalg.norm(np.abs(np.array(img[x][0][:]) - np.array(img[x][1][:])),norm)
	elif y==len(img[0][:])-1:
		e+= 2 * np.linalg.norm(np.abs(np.array(img[x][y][:]) - np.array(img[x][y-1][:])),norm)
	else:
		e+= np.linalg.norm(np.abs(np.array(img[x][y][:]) - np.array(img[x][y-1][:])),norm) + np.linalg.norm(np.abs(np.array(img[x][y][:]) - np.array(img[x][y+1][:])),norm)
	return e

#function that returns points of seam to be removed
def get_values_to_be_removed(disp,min_index):
	points=[]
	i = len(disp)-1
	while i >=0:
		points.append([i,min_index])
		min_index += disp[i][min_index]
		i-=1;
	return points

#function to get neighbors of removed points
def get_neighbors(point):
	x = point[0]
	y = point[1]
	#returning [x,y] instead of [x,y+1], because after deletion the previous [x,y+1] value is now [x.y].
	return [[x-1,y],[x,y-1],[x,y],[x+1,y]]

#function to get indices of all points that need to have their energies recomputed
def get_values_to_be_recalculated(points,energy):
	points_neighbors=[]
	for point in points:
		neighbors = get_neighbors(point)
		for candidate in neighbors:
			if(candidate[0]>=0 and candidate[0] < len(energy) and candidate[1]>=0 and candidate[1] < (len(energy[0]) - 1)):
				if candidate not in points_neighbors:
					points_neighbors.append(candidate)
	return points_neighbors

#function for deleting pixels corresponding to the lowest energy seam 
def delete_points(points,mat):
	for point in points:
		del mat[point[0]][point[1]]

#function for converting img to a list
def convert_image_to_list(img):
	img_modified=[]
	for i in range(img.shape[0]):
		temp=[]
		for j in range(img.shape[1]):
			temp.append(img[i,j,:])
		img_modified.append(temp)
	return img_modified

#function to generate energy matrix from image
def generate_energy_matrix(img):
	energy = []
	for i in range(img.shape[0]):
		temp=[]
		for j in range(img.shape[1]):
			temp.append(get_energy(img,i,j))
		if i%10==0:
			print(i ," rows processed")
		energy.append(temp)
	return energy

#function to create a mask based on input
def create_mask(rects,img):
	mask = np.ones((img.shape[0],img.shape[1]))
	for rect in rects:
		pt1=rect[0]
		pt2=rect[1]
		if rect[0][0]<rect[1][0]:
			left = rect[0][0]
			right = rect[1][0]
		else:
			left = rect[1][0]
			right = rect[0][0]
		if rect[0][1] < rect[1][1]:
			up = rect[0][1]
			down = rect[1][1]
		else:
			up = rect[1][1]
			down = rect[0][1]
		#print(up,down,left,right)
		mask[max(up,0):down,max(left,0):right] = 10
	return mask

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-f", "--file", default=1,
	help="path to text file containing intensity difference of each pixel, if already generated")
ap.add_argument("-s", "--size", default=None,
	help="size to be reduced to as a percent of current width")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
img_modified=convert_image_to_list(img)

if args["file"] is None:
	energy = generate_energy_matrix(img)
	with open("energy.txt", "wb") as fp:
		pickle.dump(energy,fp)
else:
	with open(args["file"], "rb") as fp:
		energy = pickle.load(fp)

#num_columns is the number of columns to be removed
if float(args["size"])>=1:
	num_columns = int(0.15*img.shape[1])
else:
	num_columns = int((1- float(args["size"]))*img.shape[1]) 

mat = copy.deepcopy(energy)
disp = copy.deepcopy(mat)

#repeated calls to remove multiple seams
for i in range(num_columns):
	print(i, " seams removed")
	if i>0:
		for point in points_neighbors:
			energy[point[0]][point[1]] = get_energy(img_modified,point[0],point[1])
	set_values(mat,energy,disp)
	last_row = len(mat)-1
	min_val = min(mat[last_row])
	min_index  = mat[last_row].index(min(mat[last_row]))

	points= get_values_to_be_removed(disp,min_index)
	points_neighbors = get_values_to_be_recalculated(points,energy)
	delete_points(points,img_modified)
	delete_points(points,disp)
	delete_points(points,mat)
	delete_points(points,energy)
img_final = np.asarray(img_modified)
cv2.imshow('Reduced Image', img_final)
cv2.imshow('Resized Image', cv2.resize(img,(img.shape[1]-num_columns, img.shape[0])))
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
