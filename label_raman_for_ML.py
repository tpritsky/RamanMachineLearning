# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:27:06 2018

@author: TPritsky
"""

'''label_raman_data_for_ml: Contains functions to extract and return labeled raman data from 
input files. These functions are called by ML_Raman.py to provide labeled data for machine learning.'''

#from pyshp import shapefile
import numpy as np
#import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#import pylab as pl

#from ML_Raman.py import loadMatFile

#Set file for tissue labels
tissue_label_file = r'C:\Users\TPritsky\Desktop\Intuitive_Project\Raman\20180424\20180424-RAM-001.txt'

#Import Raman Data
raman_data_file = r'C:\Users\TPritsky\Desktop\Intuitive_Project\Raman\20180424\20180424-RAM-001_VARIABLES_1'
#data, x_values, y_values, im = loadMatFile(raman_data_file)

'''Note: RamanInBounds will return the same point with multiple labels assigned if multiple labels 
are assigned to tissue'''


'''pointInPlot: determines if a given point is inside a polygon with vertices given 
    by 'vertices' '''
def pointInPlot(vertices, point):
    point = Point(point)        #convert coordinates to point object
    polygon = Polygon(vertices)     #create polygon object
    return (polygon.contains(point))    #return true if point is in polygon

'''getTissueBoundaries: read data from textfile(tissue_label_file) and return a dictionary, with key
being a tissue type (string) and value being a vector of bounding polygons for that tissue type.'''
def getTissueBoundaries(tissue_label_file):    
    #polygon dictionary: key is the tissue type and value is a vector of bounding polygons
    polygon_dict = {}
    
    #read label data
    tissue_label_data = open(tissue_label_file, "r")
    for line in tissue_label_data.readlines()[2:]:  #CHECK BOUNDS
        #create vector to store all polygons of a given tissue type
        all_polygons = []
        
        #create polygons
        tissue_label = line.split(':')[0]   #store tissue type
        tissue_label.replace(",", "_")   #ensure that there is no whitespace
        tissue_label_no_whitespace = ""                
        #print s
        print("Tissue data:")
        for i in tissue_label[:-1]:
                tissue_label_no_whitespace+=str(i)
        
        #print trial
        print(tissue_label_no_whitespace)
        
        tissue_data = line.split(':')[1]   #get all data
        for polygon in tissue_data.split('|'):   #split into distinct tissue regions
            tissue_coordinates = polygon.split('-')    #get vertices from each tissue region
            point_array = []        #an array that contains the vertices of a polygon
            for coordinate in tissue_coordinates[:-1]:  
                    i = coordinate.split(',')           #split vertices into x,y coordinates
                    #print(i[0])
                    i[0].replace(' ','')    #ensure that there is no whitespace
                    i[1].replace(' ','')
                    point = Point(float(i[0]),float(i[1]))             #create point object
                    point_array.append(point)           #append vertice point
            new_polygon = Polygon([[p.x,p.y] for p in point_array[:-1]])      #create polygon object
            all_polygons.append(new_polygon)
            
            #plotting to debug whether points were found in polygons
            '''plt.figure()
            x = [p.x for p in point_array]
            y = [p.y for p in point_array]
            
            point = Point(float(100),float(400))
            if(new_polygon.contains(point)):
                print("Point Within")
            
            plt.plot(x,y)
            plt.draw()
            plt.show()
            plt.pause(1)
            plt.show()'''
            
        #append polygons to polygon_dict. Key is tissue type and value is a list of polygons
        if not tissue_label_no_whitespace in polygon_dict:
            polygon_dict[tissue_label_no_whitespace] = all_polygons
        else:
            polygon_dict[tissue_label_no_whitespace].append(all_polygons)
    return polygon_dict     
                
'''RamanInBounds: Determine if a raman coordinates are in bounds of labeled tissue data. Input is
raman_data_x, a list of raman data x_coordinates; raman_data_y,  a list of raman data y_coordinates;
polygon_dict, a dictionary with key of tissue type and value of a list of Polygons bounding that tissue. 
Returns three vectors: tissue_types, a vector of tissue types for inputted raman data; x_coordinates, a 
vector of corresponding raman data x_coordinates; and y_coordinates, a vector of corresponding raman data
y_coordinates.'''

############################Make sure to set tissue_types values to actual tissue type###################################

def RamanInBounds(raman_data_x, raman_data_y, polygon_dict):
    tissue_types = []   #stores tissue-type labels for Raman pixels
    x_coordinates = []  #stores x_coordinates for labeled raman data
    y_coordinates = []  #stores y_coordinates for labeled raman data
    
    #iterate through all raman data points to determine which are labeled
    for i in range(len(raman_data_x)):
        
        point = Point(float(raman_data_x[i]), float(raman_data_y[i]))
        for key in polygon_dict.keys():
            for polygon in polygon_dict[key]:
                if(polygon.contains(point)):        #check if pixel is labeled
                    #print("Point Within")
                    x_coordinates.append(raman_data_x[i])   #store x_coordinate of labeled pixel
                    y_coordinates.append(raman_data_y[i])   #store y_coordinate of labeled pixel
                    tissue_types.append(key)       #store tissue label for raman pixel
    
    return (x_coordinates, y_coordinates, tissue_types)

'''labelRamanData: Function returns labeled_raman_data, a 2D numpy array with rows corresponding to 
samples and columns corresponding to features. The labeled_raman_data vector contains all samples
that had been labeled in the tissue_label_file. The inputs are 1. raman_data, an unlabeled 2D vector 
with rows corresponding to samples and columns corresponding to features; 2/3. unlabeled_x/
unlabeled_y: lists of the original raman data x/y_coordinates; 4/5. labeled_x/
labeled_y, lists of x/y_coordinates corresponding to labeled raman data.'''
def labelRamanData(raman_data, unlabeled_x, labeled_x, unlabeled_y, labeled_y):
    labeled_raman_data = [] #raman_data corresponding to pixels that were labeled
    
    #iterate through all labeled coordinates to find corresponding unlabeled coordinates
    for i in range(len(labeled_x)):
            for j in range(len(unlabeled_x)):
                if unlabeled_x[j] == labeled_x[i]:
                    if unlabeled_y[j] == labeled_y[i]:
                        labeled_raman_data.append(raman_data[j])
    
    #convert list to numpy array
    labeled_raman_data = np.array(labeled_raman_data)
    #return vector of labeled raman data
    return labeled_raman_data

     