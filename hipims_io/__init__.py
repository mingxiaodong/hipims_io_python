#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init
To do:
    initialize a package
Created on Wed Apr  1 14:56:15 2020

@author: Xiaodong Ming
"""
import pkg_resources
import numpy as np
from .InputHipims import InputHipims
from .OutputHipims import OutputHipims
from .Raster import Raster

def demo_hipims_input(set_example_inputs=True):
    """ A demonstration to generate a hipims input object
    """
    dem_file = pkg_resources.resource_filename(__name__,
                                             'sample/Example_DEM.asc')
    obj_in = InputHipims(dem_data=dem_file)
    if set_example_inputs:
        __set_defaul_input(obj_in)
    # show model summary print(obj_in)
    obj_in.Summary.display()
    fig, ax = obj_in.domain_show(relocate=True, scale_ratio=1000)
    ax.set_title('The Upper Lee catchment')
    return obj_in

def demo_raster():
    """ A demonstration to read and show raster files
    """
    dem_file = pkg_resources.resource_filename(__name__,
                                             'sample/Example_DEM.asc')
    obj_ras = Raster(dem_file)
    fig, ax = obj_ras.mapshow(relocate=True, scale_ratio=1000)
    ax.set_title('The Upper Lee catchment DEM (mAOD)')
    return obj_ras

def __set_defaul_input(obj_in):
    """Set some default values for an InputHipims object
    """
    # define initial condition
    h0 = obj_in.Raster.array+0
    h0[np.isnan(h0)] = 0
    h0[h0 < 50] = 0
    h0[h0 >= 50] = 1
    # set initial water depth (h0) and velocity (hU0x, hU0y)
    obj_in.set_parameter('h0', h0)
    obj_in.set_parameter('hU0x', h0*0.0001)
    obj_in.set_parameter('hU0y', h0*0.0002)
    # define boundary condition
    bound1_points = np.array([[535, 206], [545, 206],
                              [545, 210], [535, 210]])*1000
    bound2_points = np.array([[520, 230], [530, 230],
                              [530, 235], [520, 235]])*1000
    bound1_dict = {'polyPoints': bound1_points,
                   'type': 'open', 'h': [[0, 10], [60, 10]]}
    bound2_dict = {'polyPoints': bound2_points,
                   'type': 'open', 'hU': [[0, 50000], [60, 30000]]}
    bound_list = [bound1_dict,bound2_dict]
    obj_in.set_boundary_condition(bound_list, outline_boundary='fall')
    # define and set rainfall mask and source (two rainfall sources)
    rain_source = np.array([[0, 100/1000/3600, 0],
                            [86400, 100/1000/3600, 0],
                            [86401, 0, 0]])
    rain_mask = obj_in.Raster.array+0
    rain_mask[np.isnan(rain_mask)] = 0
    rain_mask[rain_mask < 50] = 0
    rain_mask[rain_mask >= 50] = 1
    #obj_in.set_parameter('precipitation_mask', rain_mask)
    obj_in.set_rainfall(rain_mask=rain_mask, rain_source=rain_source)
    # define and set monitor positions
    gauges_pos = np.array([[534.5, 231.3], [510.2, 224.5], [542.5, 225.0],
                           [538.2, 212.5], [530.3, 219.4]])*1000
    obj_in.set_gauges_position(gauges_pos)
    # add a user-defined parameter
#    obj_in.add_user_defined_parameter('new_param',0)