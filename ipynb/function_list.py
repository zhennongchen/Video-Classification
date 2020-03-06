#!/usr/bin/env python

# this script saved most of functions widely used in other scripts
import numpy as np
import math
import glob as gb
import glob
import os
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib
from nibabel.affines import apply_affine
import math
import xlsxwriter as xl
import string
import cv2

# function: normalize one vector
def normalize(x):
    x_scale = np.linalg.norm(x)
    return np.asarray([i/x_scale for i in x])

# function: get length of one vector and angle between two vectors
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    rad=math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    result = rad / math.pi * 180
    return result

# function: get a vector which is with a certain angle from one known vector
def vector_with_angle(v,angle):
    return np.dot(np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]),np.array([[v[0]],[v[1]]])).reshape(2,)

# function: project one vector onto a plane with known normal vectors
def project_onto_plane(u,n):
    '''n is the normal vector of the plane'''
    n = normalize(n)
    return (u - dotproduct(u,n) * n)

# function: only pick one time frame from each patient
def even(x):
  '''only pick one time frame for each patient'''
  for i in range(len(x)):
    if i%2 == 1:
      x[i]='0'
  return x[x!='0']

# function: turn normalized vector into pixel unit
def turn_to_pixel(vec,size=[160,160,96]):
    t=vec.reshape(3,).tolist()
    result = [t[i]*size[i]/2 for i in range(0,3)]
    return np.array(result)

# function: extract vectors from numpy file
def get_ground_truth_vectors(main_path,file_name):
    a = np.load(os.path.join(main_path,'affine_standard',file_name),allow_pickle=True)
    [t,x,y,s,img_center] = [a[12],a[5],a[7],a[11],a[14]]
    result = {'t':t,'x':x,'y':y,'s':s,'img_center':img_center}
    return result

def get_predicted_vectors(main_path,file1,file2,vector_true):
    f1 = np.load(os.path.join(main_path,'matrix-pred',file1),allow_pickle=True)
    f2 = np.load(os.path.join(main_path,'matrix-pred',file2),allow_pickle=True)
    t = turn_to_pixel(f1[0])
    [x,y] = [f2[1],f2[-1]]
    result = {'t':t,'x':x,'y':y,'s':vector_true['s'],'img_center':vector_true['img_center']}
    return result

# function: find a list of all center coordinate given start point and number of planes for SA stack
def find_center_list(start_center,n,num_of_plane,slice_thickness):
    # n is normal vector
    center_list = start_center.reshape(1,3)
    for i in range(1,num_of_plane):
        c = start_center + n * slice_thickness * (i) / 1.5
        center_list = np.concatenate((center_list,c.reshape(1,3)))
    return center_list

# find center list for the whole stack
def find_center_list_whole_stack(start_center,n,num_a,num_b,slice_thickness):
    # n is normal vector
    center_list = start_center.reshape(1,3)
    for i in range(1,num_a + 1):
        c = start_center + -n * slice_thickness * (i) / 1.5
        center_list = np.concatenate((c.reshape(1,3),center_list))
    for i in range(1,num_b + 1):
        c = start_center + n * slice_thickness * (i) / 1.5
        center_list = np.concatenate((center_list,c.reshape(1,3)))
    return center_list

# find center list for the whole stack with upsampled center points (default slice_thickness = 8mm which we used to calculate num_a and num_b, now we want slicethickness to be arbitrary number)
def find_center_list_whole_stack_upsampled(start_center,n,num_a,num_b,new_thickness,default_thickness = 8):
    center_list = find_center_list_whole_stack(start_center,n,num_a,num_b,default_thickness)

    # now calculate the LV length
    total_length = default_thickness * (num_a+num_b) 
    # another way to calculate total_length
    dif = center_list[-1] - center_list[0]
    total_length2 = dif[0] / normalize(dif)[0] * 1.5
    assert int(total_length) == int(total_length2)
    # the number of upsampled points
    num_p = int(total_length / new_thickness + 1) + 1
    new_center_list = center_list[0].reshape(1,3)
    for i in range(1,num_p):
        c = center_list[0] + n * new_thickness * (i) / 1.5
        new_center_list = np.concatenate((new_center_list,c.reshape(1,3)))
    return new_center_list
    
    


# function: copy image
def copy_image(image):
    if len(image.shape) == 3:
        result = np.zeros((image.shape[0],image.shape[1],1))
    else:
        result = np.zeros((image.shape[0],image.shape[1]))
    for ii in range(0,image.shape[0]):
        for jj in range(0,image.shape[1]):
            if len(image.shape) == 3: 
                result[ii,jj,0] = image[ii,jj,0]
            else:
                result[ii,jj] = image[ii,jj]
    return result
    
# function: find the affine matrix for any plane in SA stack based on the basal affine matrix
def sa_affine(plane_center,image_center,basal_vectors):
    result = {'t':plane_center - image_center,'x':basal_vectors['x'],'y':basal_vectors['y'],'s':basal_vectors['s'],'img_center':basal_vectors['img_center']}
    return result

# function: define the interpolation
def define_interpolation(data,Fill_value=0,Method='linear'):
    shape = data.shape
    [x,y,z] = [np.linspace(0,shape[0]-1,shape[0]),np.linspace(0,shape[1]-1,shape[1]),np.linspace(0,shape[-1]-1,shape[-1])]
    interpolation = RegularGridInterpolator((x,y,z),data,method=Method,bounds_error=False,fill_value=Fill_value)
    return interpolation

# function: reslice a mpr
def reslice_mpr(mpr_data,plane_center,x,y,x_s,y_s,interpolation):
    # plane_center is the center of a plane in the coordinate of the whole volume
    mpr_shape = mpr_data.shape
    new_mpr=[]
    centerpoint = np.array([(mpr_shape[0]-1)/2,(mpr_shape[1]-1)/2,0])
    for i in range(0,mpr_shape[0]):
        for j in range(0,mpr_shape[1]):
            delta = np.array([i,j,0])-centerpoint
            v = plane_center + (x*x_s)*delta[0]+(y*y_s)*delta[1]
            new_mpr.append(v)
    new_mpr=interpolation(new_mpr).reshape(mpr_shape)
    return new_mpr

# function: check affine from all time frames
def check_affine(one_time_frame_file_name):
    """this function uses the affine with each element as the mode in all time frames"""
    joinpath = os.path.join(os.path.dirname(one_time_frame_file_name),'*.nii.gz')
    f = np.array(sorted(glob.glob(joinpath)))
    a = np.zeros((4,4,len(f)))
    count = 0
    for i in f:
        i = nib.load(i)
        a[:,:,count] = i.affine
        count += 1
    mm =nib.load(f[0])
    result = np.zeros((4,4))
    for ii in range(0,mm.affine.shape[0]):
        for jj in range(0,mm.affine.shape[1]):
            l = []
            for c in range(0,len(f)):
                l.append(a[ii,jj,c])
            result[ii,jj] = max(set(l),key=l.count)
    return result

# function: get affine matrix from translation,x,y and scale
def get_affine_from_vectors(mpr_data,volume_affine,vector):
    # it answers one important question: what's [1 1 1] in the coordinate system of predicted plane in that
    # of the whole CT volume
    [t,x,y,s,i_center] = [vector['t'],vector['x'],vector['y'],vector['s'],vector['img_center']]
    shape = mpr_data.shape
    mpr_center=np.array([(shape[0]-1)/2,(shape[1]-1)/2,0])
    Transform = np.ones((4,4))
    xx = normalize(x)*s[0]
    yy = normalize(y)*s[1]
    zz = normalize(np.cross(x,y))*s[-1]
    Transform[0:3,0] = xx
    Transform[0:3,1] = yy
    Transform[0:3,2] = zz
    t_o = (i_center + t) - (mpr_center[0]*xx + mpr_center[1]*yy + mpr_center[2]*zz)
    Transform[0:3,3] = t_o
    Transform[3,:] = np.array([0,0,0,1])
    mpr_A = np.dot(volume_affine,Transform)
    return mpr_A

# function: convert coordinates to different coordinate system by affine matrix                             
def convert_coordinates(target_affine, initial_affine, r):
    affine_multiply = np.linalg.inv(target_affine).dot(initial_affine)
    return apply_affine(affine_multiply,r)

# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(gb.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: color box addition
def color_box(image,y_range = 10, x_range = 20):
    [sx,sy] = [image.shape[0],image.shape[1]]
    new_image = np.ones((sx,sy))
    for i in range(sx):
        for j in range(sy):
            new_image[i,j] = image[i,j]
    for j in range(sy-y_range,sy):
        for i in range(sx-x_range,sx):
            new_image[i,j] = new_image.max()
    return new_image

# function: draw an arbitrary axis on one image 
def draw_arbitrary_axis(image,axis,start_point,length = 500):
    '''length defines how long the axis we want to draw'''
    assert abs(axis[-1]) == 0.0 
    assert abs(start_point[-1]) == 0.0
    
    result = np.zeros((image.shape[0],image.shape[1],1))
    for ii in range(0,image.shape[0]):
        for jj in range(0,image.shape[1]):
            
            result[ii,jj,0] = image[ii,jj,0]
            

    axis = normalize(axis)
  
    for i in range(-length,length):
        [x,y] = [start_point[0] + axis[0]*i , start_point[1] + axis[1]*i]
        if x>=0 and y>=0 and x<result.shape[0] and y<result.shape[1]:
            
            result[int(x),int(y),0] = result.max()
    return result

# function: draw the intersection of two mpr planes on one plane (draw the intersection of plane 1 and 2 on plane 2)
def draw_plane_intersection(plane2_image,plane1_x,plane1_y,plane1_affine,plane2_affine,volume_affine):
    '''plane 2 is the plane in which we want to draw axis'''
    real_x = convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]+plane1_x)) - convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]))
    real_y = convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]+plane1_y)) - convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]))
    
    n1 = np.cross(real_x,real_y)
    n2 = np.array([0,0,1])
    intersect_direct = (0.5/(np.cross(n1,n2)[0])) * np.cross(n1,n2)
    
    # find one point in the intersection line
    # a plane is defined as <a,b,c> . <x-x0,y-y0,z-z0> = 0
    # ax+by+cz+(-ax0-by0-cz0) = ax+by+cz+d = 0
    # let's find one (x0,y0,z0) on the 2C plane
    
    plane1_p = convert_coordinates(plane2_affine,plane1_affine,np.array([150,100,0]))
    p = convert_coordinates(plane2_affine,plane1_affine,np.array([40,40,0]))
    d1 = -n1[0]*plane1_p[0] - n1[1]*plane1_p[1] - n1[2]*plane1_p[2]   
    d2 = 0
    u = np.cross(n1,n2)
    u_length = math.sqrt(u[0]**2+u[1]**2+u[2]**2)
    intersect_point = np.cross((d2*n1-d1*n2),u)/(u_length**2)
    result_line = draw_arbitrary_axis(plane2_image,intersect_direct,intersect_point)
    return result_line,intersect_direct,intersect_point

# function: count pixel belonged to one label
def count_pixel(seg,target_val):
  a,b,c = seg.shape
  count1 = 0; 
  p1 = []; 
  for i in range(0,a):
    for j in range(0,b):
      for k in range(0,c):
        if seg[i,j,k] == target_val:
          count1 = count1+1
          p1.append([i,j,k])
  return count1, p1

# function: excel write
def xlsx_save(filepath,result,par,index_of_par):
    '''par is parameter list such as [('Patient_Class',1),('Patient_ID',1),('Assignment',1)]'''
    workbook = xl.Workbook(filepath)
    sheet = workbook.add_worksheet()
    letter = string.ascii_uppercase

    k = 0
    for i in range(0,len(index_of_par)):
        index = index_of_par[i]
        L = letter[k] + '1'
        P = par[index][0]
        sheet.write(L,P)
        k = k + par[index][1]

    row = 1
    col = 0
    for result_list in result:
        kk = 0
        for i in range(0,len(index_of_par)):
            index = index_of_par[i]
            content = result_list[i]
            if par[index][1] == 1:
                sheet.write(row, col + kk, content)
            else:
                sheet.write_row(row, col + kk, content)
            kk = kk + par[index][1]
        row = row + 1
    workbook.close()

# function: find time frame of a file
def find_timeframe(file,num_of_dots):
    k = list(file)
    if num_of_dots == 1: #.png
        num1 = [i for i, e in enumerate(k) if e == '.'][-1]
    else:
        num1 = [i for i, e in enumerate(k) if e == '.'][-2]
    num2 = [i for i,e in enumerate(k) if e== '/'][-1]
    kk=k[num2+1:num1]
    if len(kk)>1:
        return int(kk[0])*10+int(kk[1])
    else: 
        return int(kk[0])

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots):
    time=[]
    time_s=[]
    for i in files:
        a = find_timeframe(i,num_of_dots)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

# function: set window level and width
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])

    high = level + width
    low = level - width
    # normalize
    unit = (1-0) / (width*2)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if image[i,j] > high:
                image[i,j] = high
            if image[i,j] < low:
                image[i,j] = low
            norm = (image[i,j] - (low)) * unit
            image[i,j] = norm
    return image


# function: decomposite a quaterninan matrix
def decompositeQ(Q):
    a = math.sqrt(1 - Q[0]**2 - Q[1]**2 - Q[-1]**2)
    a_rad = math.acos(a) * 2
    a = a_rad / math.pi * 180
    print('angle',a)
    sin = math.sin(a_rad/2)
    [ux,uy,uz] = [Q[0] / sin, Q[1] / sin, Q[-1] / sin]
    u = np.array([ux,uy,uz])
    square_sum = ux**2+uy**2+uz**2
    print(square_sum)
    return a,u,square_sum

# function: find aha segments vectors from two RV insertion points and the LV center
def find_aha_segments(c,i12,i34):
    '''c,i12 and i34 are coordinates of LV center and two RV insertion points saved in numpy file.
    this function can then use saved information to get the vectors (v12 to v61) representing 6 AHA segments in MID plane'''
    v12 = normalize(i12 - c)
    v34 = normalize(i34 - c)
    
    septum_angle = math.acos(dotproduct(v12, v34) / (length(v12) * length(v34)))/2
    lateral_angle =  (2*math.pi - septum_angle*2) / 4

    v61 = np.dot(np.array([[math.cos(lateral_angle),-math.sin(lateral_angle)],[math.sin(lateral_angle),math.cos(lateral_angle)]]),np.array([[v12[0]],[v12[-1]]])).reshape(2,)
    v56 = np.dot(np.array([[math.cos(lateral_angle),-math.sin(lateral_angle)],[math.sin(lateral_angle),math.cos(lateral_angle)]]),np.array([[v61[0]],[v61[-1]]])).reshape(2,)
    v45 = np.dot(np.array([[math.cos(lateral_angle),-math.sin(lateral_angle)],[math.sin(lateral_angle),math.cos(lateral_angle)]]),np.array([[v56[0]],[v56[-1]]])).reshape(2,)
    v34 = np.dot(np.array([[math.cos(lateral_angle),-math.sin(lateral_angle)],[math.sin(lateral_angle),math.cos(lateral_angle)]]),np.array([[v45[0]],[v45[-1]]])).reshape(2,)
    v23 = np.dot(np.array([[math.cos(septum_angle),-math.sin(septum_angle)],[math.sin(septum_angle),math.cos(septum_angle)]]),np.array([[v34[0]],[v34[-1]]])).reshape(2,)
    
    v61 = np.array([v61[0],v61[1],0]);v12 = np.array([v12[0],v12[1],0]);v23 = np.array([v23[0],v23[1],0]);v34 = np.array([v34[0],v34[1],0]);v45 = np.array([v45[0],v45[1],0]);v56 = np.array([v56[0],v56[1],0])
    result = np.array([v12,v23,v34,v45,v56,v61])
    return result

# function: draw aha segments on the MID
def draw_aha_segments(image,AHA_axis,LV_center):
    result = copy_image(image)
    
    # load 6 vectors for segments from axis
    vectors = []
    for a in AHA_axis:
        vectors.append(normalize(a))
    c = np.array([LV_center[0],LV_center[1],0])
    
    # draw lines
    for v in vectors:
        for i in range(0,50):
            [x,y] = [c[0] + v[0]*i , c[1] + v[1]*i]
            if x>0 and y>0 and x <image.shape[0] and y <image.shape[1]:
                result[int(x),int(y),0] = result.max() 
    return result

# function: find the number of slices upper the basal plane and the number of slices lower the basal plane for SAX stack. the stack starts from 2 planes before where
# LV segmentation starts and ends two planes after where LV segmentaion ends. 
def find_num_of_slices_in_SAX(mpr_data,image_center,t_m,x_m,y_m,seg_m_data,slice_thickness = 8):
    n_m =  normalize(np.cross(x_m,y_m))

    test_a = 1
    a_manual = 0
    while test_a == True:
        a_manual += 1
        plane = reslice_mpr(mpr_data,(image_center + t_m + (-n_m) * slice_thickness * a_manual / 1.5),x_m,y_m,1,1,define_interpolation(seg_m_data,Fill_value=0,Method='nearest'))
        test_a = (1.0 in plane)
    a_manual += 1

    test_b = 1
    b_manual = 0
    while test_b == True:
        b_manual += 1
        plane = reslice_mpr(mpr_data,(image_center + t_m + (n_m) * slice_thickness * b_manual / 1.5),x_m,y_m,1,1,define_interpolation(seg_m_data,Fill_value=0,Method='nearest'))
        test_b = (1.0 in plane)
    b_manual += 1
    
    return a_manual,b_manual


# function: find which plane is base, mid or apex
def particular_plane_in_stack(a,b,num_of_section,base_no,mid_no,apex_no):
    '''a and b's meaning can be found in function find_num_of_slices_in_SAX'''
    start = 3
    end = a + b +1 - 2

    gap = (end - start ) / num_of_section

    base = ((gap * base_no) ) + start
    mid = ((gap * mid_no) ) + start
    apex = ((gap * apex_no) ) + start
    return start,end,base,mid,apex

# function: upsample image
def upsample_images(image,up_size = 1):
    # in case it's 2D image
    if len(image.shape) == 2:
        image = image.reshape(image.shape[0],image.shape[1],1)
    # interpolation by RegularGridInterpolator only works for images with >1 slices, so we need to copy the current slice
    I = np.zeros((image.shape[0],image.shape[1],2))
    I[:,:,0] = image[:,:,0];I[:,:,1] = image[:,:,0]
    # define interpolation
    interpolation = define_interpolation(I,Fill_value=I.min(),Method='linear')
    
    new_image = []
    new_size = [image.shape[0]*up_size,image.shape[1]*up_size]
    
    for i in range(0,new_size[0]):
        for j in range(0,new_size[1]):
            point = np.array([1/up_size*i,1/up_size*j,0])
            new_image.append(point)
            
    new_image = interpolation(new_image).reshape(new_size[0],new_size[1],1)
    return new_image

# function: make movies
def make_movies(save_path,pngs,timeframe_rate = 6):
    '''pngs are all file paths of pngs used to generate movie'''
    mpr_array=[]
    for j in pngs:
        img = cv2.imread(j)
        h,w,l = img.shape
        size = (w,h)
        mpr_array.append(img)

    # save movies
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'MJPG'),timeframe_rate,size)
    for j in range(len(mpr_array)):
        out.write(mpr_array[j])
    out.release()
    