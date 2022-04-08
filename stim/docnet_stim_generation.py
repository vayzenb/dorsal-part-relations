import bpy 
import random
import numpy as np
import os
from glob import glob
from math import radians
import time
import pdb
import sys
import pdb

#terminal commands: 
#/lab_data/hawk/blender/blender-2.93.2/blender -b docnet_image_creation.blend -P docnet_stim_generation.py
bpy.context.scene.render.engine = 'CYCLES'
curr_dir = '/home/vayzenbe/GitHub_Repos/docnet/stim'
os.chdir(curr_dir)
model_dir = '/lab_data/behrmannlab/image_sets/ShapeNetCore.v2'
out_dir= '/lab_data/behrmannlab/image_sets/ShapeNet_images'
#im_dir = f'{curr_dir}/obj_images'

#Start orient
angle_increments = 30
min_angle = (-75)
max_angle = 75

size_inc = .1

min_size = .30
max_size = .35
max_length = 1.1

#load class name from python
#it's arg 5 & 6, because all the blender stuff is 1-4
cl = sys.argv[5]
#how many objects to use for each class
num_obj = int(sys.argv[6])


#Load model list
#model_list = np.loadtxt(f'{curr_dir}/model_list.csv', delimiter=',', dtype=object)


def create_object(obn, ob):
    obj_name = ob.split('/')[-1]

    #unselect everything
    bpy.ops.object.select_all(action='DESELECT')

    #import object model
    imported_object = bpy.ops.import_scene.obj(filepath=f'{ob}/models/model_normalized.obj')
    print('object loaded')

    '''
    #Set file path for the render
    bpy.context.scene.render.filepath = f'{curr_dir}/obj_images/test.jpg'

    #Take the picture
    bpy.ops.render.render(write_still = True)
    '''
    #set current object to variable 
    curr_obj = bpy.context.selected_objects[0]

    #select/activate it
    curr_obj.select_set(True)
    bpy.context.view_layer.objects.active = curr_obj

    #move and change scale if relevant
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')

    bpy.context.object.location = [0, 0, 0]
    
    
    curr_inc = 1.0
    if (bpy.context.object.dimensions[0]*bpy.context.object.dimensions[1]) < min_size:

        while (bpy.context.object.dimensions[0]*bpy.context.object.dimensions[1]) < min_size:
            curr_inc = float(curr_inc + size_inc)
            bpy.context.object.scale = [curr_inc, curr_inc, curr_inc]
            bpy.context.view_layer.update()

            if bpy.context.object.dimensions[0] > max_length or bpy.context.object.dimensions[1] > max_length or bpy.context.object.dimensions[2] > max_length:
                break
            
    elif (bpy.context.object.dimensions[0]*bpy.context.object.dimensions[1]) > max_size:
        while (bpy.context.object.dimensions[0]*bpy.context.object.dimensions[1]) > max_size:
            curr_inc = float(curr_inc - size_inc)
            bpy.context.object.scale = [curr_inc, curr_inc, curr_inc]
            bpy.context.view_layer.update()
            
            if curr_inc <= .6:
                break

    #Remove the material from the object
    for mat_n, material in enumerate(bpy.data.materials):
        # clear and remove the old material
        #print(material)
        material.user_clear()
        bpy.data.materials.remove(material)
        #print(mat_n)

        
    ob = bpy.context.active_object
    for mn in range(0, len(ob.data.materials)):
        
        bpy.context.object.active_material_index = mn
        mat = bpy.data.materials.new(name=f"Material_{mn}")
        mat.diffuse_color = (.5, 0, 0,1) 
        
        #try:
        # assign to 1st material slot
        
        ob.data.materials[mn] = mat
        
        #ob.data.materials.append(mat)

    #break
        #except:
            #continue

        #bpy.context.object.active_material.use_nodes = True
        bpy.data.materials[f"Material_{mn}"].specular_color = (0, 1, 0.5)

        #   #select that index and add a new material
        #  bpy.context.object.active_material_index = mn
        # bpy.ops.material.new()
        
    #rotate object
    rand_rot = random.randint(min_angle,max_angle)
    #db.set_trace()

    bpy.context.object.rotation_euler.z = radians(rand_rot)
    #bpy.context.object.rotation_euler.z = radians(random.randint(-65,65))

    #Set file path for the render
    #pdb.set_trace()
    bpy.context.scene.render.filepath = f'{out_dir}/{cl.split("/")[-1]}/{obj_name}.jpg'

    #Take the picture
    bpy.ops.render.render(write_still = True)



    #Delete selected object
    bpy.ops.object.delete()


#create image directory for object class
os.makedirs(f'{out_dir}/{cl.split("/")[-1]}', exist_ok = True)

#load all folders in class folder
exemplar_list = glob(f'{cl}/*')

#shuffle exemplar list
random.shuffle(exemplar_list)

#loop through objects in class folder
for obn, ob in enumerate(exemplar_list[:num_obj]):  
    print(ob)
    try:
        create_object(obn, ob)
    except:
        continue

