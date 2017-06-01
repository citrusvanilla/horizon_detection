from __future__ import division

import numpy as np
import cv2
import math


def get_plane_indicator_coord(img,angle,dist_as_perc,buffer_size):

    heading = angle + 90
    heading_from_hor = min(math.radians(180-heading),math.radians(heading))
    
    radius = int(math.ceil(math.sqrt((img.shape[1]*0.5)**2 + (img.shape[0]*0.5)**2)))
    
    x0 = img.shape[1]*0.5
    y0 = img.shape[0]*0.5
    
    x1 = (img.shape[1]*0.5) - radius
    x2 = (img.shape[1]*0.5) + radius
    
    y1 = (img.shape[0]*0.5)
    y2 = (img.shape[0]*0.5)

    x1_n = x0 + math.cos(math.radians(angle)) * (x1 - x0) - math.sin(math.radians(angle)) * (y1 - y0)
    y1_n = y0 + math.sin(math.radians(angle)) * (x2 - x0) + math.cos(math.radians(angle)) * (y2 - y0)
    
    x2_n = x0 + math.cos(math.radians(angle)) * (x2 - x0) - math.sin(math.radians(angle)) * (y2 - y0)
    y2_n = y0 + math.sin(math.radians(angle)) * (x1 - x0) + math.cos(math.radians(angle)) * (y1 - y0)

    
    if heading_from_hor == 0:
        avail_dist = img.shape[1]
    elif abs(heading_from_hor) < math.atan(img.shape[0]/img.shape[1]):
        avail_dist = abs(int(img.shape[1]/math.cos(math.radians(heading))))
    else: #heading_from_hor >= np.arctan(img.shape[0]/img.shape[1])
        avail_dist = abs(int(img.shape[0]/math.sin(math.radians(heading))))

    origin = avail_dist *0.5
    heading_transform = (dist_as_perc * avail_dist) - origin
    sky_buffer_transform = heading_transform + buffer_size
    sea_buffer_transform = heading_transform - buffer_size
    
    x_transform = heading_transform * math.cos(math.radians(heading))
    y_transform = heading_transform * math.sin(math.radians(heading))

    pos_x_transform = sky_buffer_transform * math.cos(math.radians(heading))
    pos_y_transform = sky_buffer_transform * math.sin(math.radians(heading))
    
    neg_x_transform = sea_buffer_transform * math.cos(math.radians(heading))
    neg_y_transform = sea_buffer_transform * math.sin(math.radians(heading))  
    
    return [ (int(x1_n+pos_x_transform),int(y1_n+pos_y_transform)), 
              (int(x2_n+pos_x_transform),int(y2_n+pos_y_transform)),
             (int(x1_n+x_transform),int(y1_n+y_transform)), 
              (int(x2_n+x_transform),int(y2_n+y_transform)),
             (int(x1_n+neg_x_transform),int(y1_n+neg_y_transform)), 
              (int(x2_n+neg_x_transform),int(y2_n+neg_y_transform))]


def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end.
    From http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm#Python.
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
        
    return points


def get_local_objective_buffer_means(img,plane_coordinates,angle,buffer_size):

    line_pixels = np.array(get_line(plane_coordinates[0],plane_coordinates[1]))
    origin = plane_coordinates[0]

    pos_x_transform = int(buffer_size * math.cos(math.radians(angle+90)))
    pos_y_transform = int(buffer_size * math.sin(math.radians(angle+90)))   
    neg_x_transform = int(buffer_size * math.cos(math.radians(angle-90)))
    neg_y_transform = int(buffer_size * math.sin(math.radians(angle-90)))    
    
    pos_buffer_pixels = np.array(get_line((0,0),(pos_x_transform,pos_y_transform)))
    neg_buffer_pixels = np.array(get_line((0,0),(neg_x_transform,neg_y_transform)))
    
    local_pos_mat = 0
    local_pos_count = 0
    local_neg_mat = 0
    local_neg_count = 0
    
    for pixel in line_pixels:
        relevant_pos_pixels = pixel - pos_buffer_pixels #(x,y)
        for i in relevant_pos_pixels:

            if 0 < i[0] < img.shape[1] and 0 < i[1] < img.shape[0]:

                local_pos_mat += img[(i[1],i[0])]
                local_pos_count += 1

        relevant_neg_pixels = pixel - neg_buffer_pixels
        for j in relevant_neg_pixels:
            if 0 < j[0] < img.shape[1] and 0 < j[1] < img.shape[0]:
                local_neg_mat += img[(j[1],j[0])]
                local_neg_count += 1

    return int(local_pos_mat/local_pos_count), int(local_neg_mat/local_neg_count)


def main(img_file, img_reduction, angles, distances, buffer_size, local_objective = 0):
    '''
    # 0 pos_half_mean  --> should have higher values
    # 1 neg_half_mean  --> should have lower values
    # 2 pos_half_var   --> should have low values
    # 3 neg_half_var   --> should have low values
    # 4 pos_local_mean --> should have higher values
    # 5 pos_local_mean --> should have low values
    '''

    img = cv2.imread(img_file,0)
    img = cv2.resize(img, dsize = None, fx = img_reduction, fy = img_reduction)
    vals = np.zeros((len(range(*angles)),
                     len(range(*distances)),
                     8)) #rows, columns
    i = 0
    
    for angle in xrange(*angles):
        #print(angle)
        j = 0
        for distance in xrange(*distances):
            
            points = get_plane_indicator_coord(img, angle, distance/100, buffer_size)
            endpoints = points[2:4]
            
            global_pos_mat = np.empty(0,dtype=np.uint8)
            global_neg_mat = np.empty(0,dtype=np.uint8) 

            local_pos_points = points[0:4]
            local_neg_points = points[2:6]

            for x_coor in xrange(0,img.shape[1]):
                
                if x_coor < endpoints[0][0]:
                    if angle < 0:
                        global_neg_mat = np.append(global_neg_mat,img[:,x_coor])
                    else: # angle > 0
                        global_pos_mat = np.append(global_pos_mat, img[:,x_coor])  
                
                elif x_coor >= endpoints[0][0] and x_coor < endpoints[1][0]:
                    y_d = int(endpoints[0][1] - ((x_coor-endpoints[0][0]) * np.tan(math.radians(angle))))
                    
                    if y_d < 0:
                        global_neg_mat = np.append(global_neg_mat,img[:,x_coor])
                    elif y_d > img.shape[0]:
                        global_pos_mat = np.append(global_pos_mat,img[:,x_coor])
                    else:
                        global_pos_mat = np.append(global_pos_mat, img[0:y_d,x_coor] )
                        global_neg_mat = np.append(global_neg_mat, img[y_d:img.shape[0], x_coor])
                
                else: #elif x_coor >= endpoints[1][0]:
                    if angle < 0:
                        global_pos_mat = np.append(global_pos_mat, img[:,x_coor])
                    else: # angle > 0
                        global_neg_mat = np.append(global_neg_mat,img[:,x_coor])

            if len(global_pos_mat)==0:
                vals[i,j,0]= 0
                vals[i,j,2] = 0
            else:
                vals[i,j,0] = int(np.mean(global_pos_mat))
                vals[i,j,2] = int(np.var(global_pos_mat)) 
            
            if len(global_neg_mat)==0:
                vals[i,j,1]= 0
                vals[i,j,3] = 0
            else:
                vals[i,j,1]= int(np.mean(global_neg_mat))
                vals[i,j,3]= int(np.var(global_neg_mat))
            
            if local_objective == 1:
                vals[i,j,4], vals[i,j,5] = get_local_objective_buffer_means(img,endpoints,angle,buffer_size)
            
            vals[i,j,6] = angle
            vals[i,j,7] = distance

            j+=1
            
        i += 1

    return vals