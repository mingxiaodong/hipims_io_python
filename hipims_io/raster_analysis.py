#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raster_analysis
To do:
    Read, write and analyze gridded Raster data
Created on Tue Mar 31 16:20:13 2020

@author: Xiaodong Ming
"""
import os
import sys
import copy
import gzip
import math
import numpy as np
from osgeo import ogr
from scipy import interpolate
import grid_show as gs
#%% *******************************To deal with raster data********************
#   ***************************************************************************    
class Raster(object):
    """    
    To deal with raster data with a ESRI ASCII or GTiff format
    Properties:
        source_file: file name to read grid data
        output_file: file name to write a raster object
        array: a numpy array storing grid cell values
        header: a dict storing reference information of the grid
        extent: a tuple storing outline limits of the raster (left, right, 
        bottom, top)
        extent_dict: a dictionary storing outline limits of the raster
        projection: (string) the Well-Known_Text (wkt) projection information
    
    Methods(public):
        Write_asc: write grid data into an asc file with or without 
            compression(.gz)
        To_osgeo_raster: convert this object to an osgeo raster object
        rect_clip: clip raster according to a rectangle extent
        clip: clip raster according to a polygon
        rasterize: rasterize a shapefile on the Raster object and return a 
            bool array with 'Ture' in and on the polygon/polyline
        resample: resample the raster to a new cellsize
        GetXYcoordinate: Get X and Y coordinates of all raster cells
        GetGeoTransform: Get GeoTransform tuple for osgeo raster dataset
        mapshow: draw a map of the raster dataset
        VelocityShow: draw velocity vectors as arrows with values on two Raster
            datasets (u, v)
            
    Methods(private):
        __header2extent: convert header to extent
        __read_asc: read an asc file ends with .asc or .gz
        __map2sub: convert map coordinates of points to subscripts of a matrix
            with a reference header
        __sub2map: convert subscripts of a matrix to map coordinates
        __read_tif: read tiff file
        
    """
#%%======================== initialization function ===========================   
    def __init__(self, source_file=None, array=None, header=None, 
                 epsg=None, projection=None, num_header_rows=6):
        """
        source_file: name of a asc/tif file if a file read is needed
        array: values in each raster cell [a numpy array]
        header: georeference of the raster [a dictionary containing 6 keys]:
            nrows, nclos [int]
            cellsize, xllcorner, yllcorner
            NODATA_value
        epsg: epsg code [int]
        projection: WktProjection [string]
        """
        self.source_file = source_file
        self.projection = projection
        if epsg is not None:
            self.projection = self.__SetWktProjection(epsg)
        else:
            self.projection = None
        if source_file is None:
            self.array = array
            self.header = header
            self.source_file = 'source_file.asc'
        elif type(source_file) is str:
            if os.path.exists(source_file):
                if source_file.endswith('.tif'):
                    self.__read_tif() # only read the first band
                else:
                    self.__read_asc(num_header_rows)
            else:
                raise IOError 
                sys.exit(1)
        else:  #try a binary file-like object
            self.__read_bytes()

        if isinstance(self.header, dict)==0:
            raise ValueError('header is not a dictionary')
        else:
            # create self.extent and self.extent_dict 
            self.__header2extent()
            
#%%============================= Spatial analyst ==============================   
    def rect_clip(self, clipExtent):
        """
        clipExtent: left, right, bottom, top
        clip raster according to a rectangle extent
        return:
           a new raster object
        """
        new_obj = copy.deepcopy(self)
        X = clipExtent[0:2]
        Y = clipExtent[2:4]
        rows, cols = self.__map2sub(X, Y)
        Xcentre, Ycentre = self.__sub2map(rows, cols)
        xllcorner = min(Xcentre)-0.5*self.header['cellsize']
        yllcorner = min(Ycentre)-0.5*self.header['cellsize']
        # new array
        new_obj.array = self.array[min(rows):max(rows), min(cols):max(cols)]
        # new header
        new_obj.header['nrows'] = new_obj.array.shape[0]
        new_obj.header['ncols'] = new_obj.array.shape[1]
        new_obj.header['xllcorner'] = xllcorner
        new_obj.header['yllcorner'] = yllcorner
        # new extent
        new_obj.__header2extent()
        new_obj.source_file = None       
        return new_obj
    
    def clip(self, mask=None):
        """
        clip raster according to a mask
        mask: 
            1. string name of a shapefile
            2. numpy vector giving X and Y coords of the mask points
        
        return:
            a new raster object
        """
        if isinstance(mask, str):
            shpName =  mask
        # Open shapefile datasets        
        shpDriver = ogr.GetDriverByName('ESRI Shapefile')
        shpDataset = shpDriver.Open(shpName, 0) # 0=Read-only, 1=Read-Write
        layer = shpDataset.GetLayer()
        shpExtent = np.array(layer.GetExtent()) #(minX, maxY, maxX, minY)           
        # 1. rectangle clip raster
        new_obj = self.rect_clip(shpExtent)
        new_raster = copy.deepcopy(new_obj)                
        indexArray = new_raster.rasterize(shpDataset)
        arrayClip = new_raster.array
        arrayClip[indexArray==0]=new_raster.header['NODATA_value']
        new_raster.array = arrayClip        
        shpDataset.Destroy()
        return new_raster
    
    def rasterize(self, shpDSName, rasterDS=None):
        """
        rasterize the shapefile to the raster object and return a bool array
            with Ture value in and on the polygon/polyline
        shpDSName: string for shapefilename, dataset for ogr('ESRI Shapefile')
            object
        return numpy array
        """
        from osgeo import gdal, ogr
        if isinstance(shpDSName, str):
            shpDataset = ogr.Open(shpDSName)
        else:
            shpDataset = shpDSName
        layer = shpDataset.GetLayer()
        if rasterDS is None:
            obj_raster = copy.deepcopy(self)
            obj_raster.array = np.zeros(obj_raster.array.shape)
            target_ds = obj_raster.To_osgeo_raster()
        else:
            target_ds = rasterDS
        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[-1])
        rasterized_array = target_ds.ReadAsArray()
        indexArray = np.full(rasterized_array.shape, False)
        indexArray[rasterized_array==-1] = True
        target_ds=None
        return indexArray
    
    def resample(self, newCellsize, method='bilinear'):
        """
        resample the raster to a new cellsize
        newCellsize: cellsize of the new raster
        method: Resampling method to use. Available methods are:
            near: nearest neighbour resampling (default, fastest algorithm, 
                                                worst interpolation quality).        
            bilinear: bilinear resampling.        
            cubic: cubic resampling.        
            cubicspline: cubic spline resampling.        
            lanczos: Lanczos windowed sinc resampling.        
            average: average resampling, computes the average of all 
                    non-NODATA contributing pixels.        
            mode: mode resampling, selects the value which appears most often 
                    of all the sampled points.        
            max: maximum resampling, selects the maximum value from all 
                    non-NODATA contributing pixels.        
            min: minimum resampling, selects the minimum value from all 
                    non-NODATA contributing pixels.        
            med: median resampling, selects the median value of all 
                    non-NODATA contributing pixels.        
            q1: first quartile resampling, selects the first quartile 
                value of all non-NODATA contributing pixels.        
            q3: third quartile resampling, selects the third quartile 
                value of all non-NODATA contributing pixels
        """
        cellSize = self.header['cellsize']
        rasterXSize = self.header['ncols']
        newRasterXSize = int(rasterXSize*cellSize/newCellsize)
        rasterYSize = self.header['nrows']
        newRasterYSize = int(rasterYSize*cellSize/newCellsize)
        
        from osgeo import gdal
        g = self.To_osgeo_raster() # get original gdal dataset
        total_obs = g.RasterCount
        drv = gdal.GetDriverByName( "MEM" )
        dst_ds = drv.Create('', g.RasterXSize, g.RasterYSize, 1,
                            eType=gdal.GDT_Float32)
        dst_ds.SetGeoTransform( g.GetGeoTransform())
        dst_ds.SetProjection ( g.GetProjectionRef() )
        hires_data = self.array
        dst_ds.GetRasterBand(1).WriteArray ( hires_data )
        
        geoT = g.GetGeoTransform()
        drv = gdal.GetDriverByName( "MEM" )
        resampled_ds = drv.Create('', newRasterXSize, newRasterYSize, 1, 
                                  eType=gdal.GDT_Float32)

        newGeoT = (geoT[0], newCellsize, geoT[2],
                   geoT[3], geoT[3], -newCellsize)
        resampled_ds.SetGeoTransform(newGeoT )
        resampled_ds.SetProjection (g.GetProjectionRef() )
        resampled_ds.SetMetadata ({"TotalNObs":"%d" % total_obs})

        gdal.RegenerateOverviews(dst_ds.GetRasterBand(1),
                                 [resampled_ds.GetRasterBand(1)], method)
    
        resampled_ds.GetRasterBand(1).SetNoDataValue(self.header['NODATA_value'])
        
        new_obj = self.__osgeoDS2raster(resampled_ds)
        resampled_ds = None

        return new_obj
    
    def Interpolate_to(self, points, values, method='nearest'):
        """ Interpolate values of 2D points to all cells on the Raster object
        2D interpolate
        points: ndarray of floats, shape (n, 2)
            Data point coordinates. Can either be an array of shape (n, 2), 
            or a tuple of ndim arrays.
        values: ndarray of float or complex, shape (n, )
            Data values.
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            Method of interpolation.
        """
        grid_x, grid_y = self.GetXYcoordinate()
        array_interp = interpolate.griddata(points, values, (grid_x, grid_y),
                                            method=method)
        new_obj = copy.deepcopy(self)
        new_obj.array = array_interp
        new_obj.source_file = 'mask_'+new_obj.source_file
        return new_obj
    
    def grid_interpolate(self, value_grid, method='nearest'):
        """ Interpolate values of a grid to all cells on the Raster object
        2D interpolate
        value_grid: a grid file string or Raster object 
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            Method of interpolation.
        Return: 
            a numpy array with the same size of the self object
        """
        if type(value_grid) is str:
            value_grid = Raster(value_grid)
        points_x, points_y = value_grid.GetXYcoordinate()
        points = np.c_[points_x.flatten(), points_y.flatten()]
        values = value_grid.array.flatten()
        ind_nan = ~np.isnan(values)
        grid_x, grid_y = self.GetXYcoordinate()
        array_interp = interpolate.griddata(points[ind_nan, :], values[ind_nan],
                                            (grid_x, grid_y), method=method)
        return array_interp
    
    def GridResample(self, newsize):
        """
        resample a grid to a new grid resolution via nearest interpolation
        """
        zMat = self.array
        header = self.header
        if isinstance(newsize, dict):
            head_new = newsize.copy()
        else:            
            head_new = header.copy()
            head_new['cellsize'] = newsize
            ncols = math.floor(header['cellsize']*header['ncols']/newsize)
            nrows = math.floor(header['cellsize']*header['nrows']/newsize)
            head_new['ncols']=ncols
            head_new['nrows']=nrows
        #centre of the first cell in zMat
        x11 = head_new['xllcorner']+0.5*head_new['cellsize']
        y11 = head_new['yllcorner']+(head_new['nrows']-0.5)*head_new['cellsize']
        xAll = np.linspace(x11,x11+(head_new['ncols']-1)*head_new['cellsize'],head_new['ncols'])
        yAll = np.linspace(y11,y11-(head_new['nrows']-1)*head_new['cellsize'],head_new['nrows'])
        rowAll,colAll = self.__map2sub(xAll,yAll)
        rows_Z,cols_Z = np.meshgrid(rowAll,colAll) # nrows*ncols array
        zNew = zMat[rows_Z,cols_Z]
        zNew = zNew.transpose()
        zNew = zNew.astype(zMat.dtype)
#        extent_new = header2extent(head_new)
        new_obj = Raster(array=zNew, header=head_new)
        return new_obj
    
    def assign_to(self, new_header):
        """ Assign_to the object to a new grid defined by new_header 
        If cellsize are not equal, the origin Raster will be firstly 
        resampled to the target grid.
        obj_origin, obj_target: Raster objects
        """
        obj_origin = copy.deepcopy(self)
        if obj_origin.header['cellsize'] != new_header['cellsize']:
            obj_origin = obj_origin.GridResample(new_header['cellsize'])
        grid_x, grid_y = obj_origin.GetXYcoordinate()
        rows, cols = _map2sub(grid_x, grid_y, new_header)
        ind_r = np.logical_and(rows >= 0, rows <= new_header['nrows']-1)
        ind_c = np.logical_and(cols >= 0, cols <= new_header['ncols']-1)
        ind = np.logical_and(ind_r, ind_c)
#        ind = np.logical_and(ind, ~np.isnan(obj_origin.array))
        array = obj_origin.array[ind]
        array = np.reshape(array, (new_header['nrows'], new_header['ncols']))
#        array[rows[ind], cols[ind]] = obj_origin.array[ind]
        obj_output = Raster(array=array, header=new_header)
        return obj_output
         
#%%=============================Visualization==================================
    #%% draw inundation map with domain outline
    def mapshow(self, **kwargs):
        """
        Display raster data without projection
        figname: the file name to export map, if figname is empty, then
            the figure will not be saved
        figsize: the size of map
        dpi: The resolution in dots per inch
        vmin and vmax define the data range that the colormap covers
        figname=None, figsize=None, dpi=300, vmin=None, vmax=None, 
                cax=True, dem_array=None, relocate=False, scale_ratio=1
        """
        fig, ax = gs.mapshow(raster_obj=self, **kwargs)
        return fig, ax
    
    def rankshow(self, **kwargs):
        """ Display water depth map in a range defined by (d_min, d_max)
        """
        fig, ax = gs.rankshow(self, **kwargs)
        return fig, ax
    
    def hillshade(self, **kwargs):
        """ Draw a hillshade map
        """
        fig, ax = gs.hillshade(self, **kwargs)
        return fig, ax

    #%% draw velocity (vector) map
    def vectorshow(self, obj_y, **kwargs):
        """
        plot velocity map of U and V, whose values stored in two raster
        objects seperately
        """
        fig, ax = gs.vectorshow(self, obj_y, **kwargs)
        return fig, ax
#%%===============================output=======================================
    
    def GetXYcoordinate(self):
        """ Get X and Y coordinates of all raster cells
        return xv, yv numpy array with the same size of the raster object
        """
        ny, nx = self.array.shape
        cellsize = self.header['cellsize']
        # coordinate of the centre on the top-left pixel
        x00centre = self.extent_dict['left'] + cellsize/2
        y00centre = self.extent_dict['top'] - cellsize/2
        x = np.arange(x00centre, x00centre+cellsize*nx, cellsize)
        y = np.arange(y00centre, y00centre-cellsize*ny, -cellsize)
        xv, yv = np.meshgrid(x, y)
        return xv, yv
    
    def GetGeoTransform(self):
        """
        get GeoTransform tuple for osgeo raster dataset
        """
        GeoTransform = (self.extent_dict['left'], self.header['cellsize'], 0.0, 
                        self.extent_dict['top'], 0.0, -self.header['cellsize'])
        return GeoTransform
    
    def Write_asc(self, output_file, EPSG=None, compression=False):
        
        """
        write raster as asc format file 
        output_file: output file name
        EPSG: epsg code, if it is given, a .prj file will be written
        compression: logic, whether compress write the asc file as gz
        """
        if compression:
            if not output_file.endswith('.gz'):
                output_file=output_file+'.gz'        
        self.output_file = output_file
        array = self.array+0
        header = self.header
        array[np.isnan(array)]= header['NODATA_value']
        if not isinstance(header, dict):
            raise TypeError('bad argument: header')
                     
        if output_file.endswith('.gz'):
            f = gzip.open(output_file, 'wb') # write compressed file
        else:
            f = open(output_file, 'wb')
        f.write(b"ncols    %d\n" % header['ncols'])
        f.write(b"nrows    %d\n" % header['nrows'])
        f.write(b"xllcorner    %g\n" % header['xllcorner'])
        f.write(b"yllcorner    %g\n" % header['yllcorner'])
        f.write(b"cellsize    %g\n" % header['cellsize'])
        f.write(b"NODATA_value    %g\n" % header['NODATA_value'])
        np.savetxt(f, array, fmt='%g', delimiter=' ')
        f.close()
        if EPSG is not None:
            self.__SetWktProjection(EPSG)
        # if projection is defined, write .prj file for asc file

        if output_file.endswith('.asc'):
            if self.projection is not None:
                prj_file=output_file[0:-4]+'.prj'
                wkt = self.projection
                with open(prj_file, "w") as prj:        
                    prj.write(wkt)
        return None
    
    # convert this object to an osgeo raster object
    def To_osgeo_raster(self, filename=None, fileformat = 'GTiff', destEPSG=27700):        
        """
        convert this object to an osgeo raster object, write a tif file if 
            necessary
        filename: the output file name, if it is given, a tif file will be produced
        fileformat: GTiff or AAIGrid
        destEPSG: the EPSG projection code default: British National Grid
        
        return:
            an osgeo raster dataset
            or a tif filename if it is written
        """
        from osgeo import gdal, osr
        if filename is None:
            dst_filename = ''
            driverName = 'MEM'
        else:
            dst_filename = filename
            driverName = fileformat
        if not dst_filename.endswith('.tif'):
            dst_filename = dst_filename+'.tif'
    
        # You need to get those values like you did.
        PIXEL_SIZE = self.header['cellsize']  # size of the pixel...        
        x_min = self.extent[0] # left  
        y_max = self.extent[3] # top
        dest_crs = osr.SpatialReference()
        dest_crs.ImportFromEPSG(destEPSG)
        # create dataset with driver
        driver = gdal.GetDriverByName(driverName)
        ncols = int(self.header['ncols'])
        nrows = int(self.header['nrows'])
#        print('ncols:', type(ncols), ' - nrows:'+type(nrows))
        dataset = driver.Create(dst_filename, 
            xsize=ncols, 
            ysize=nrows, 
            bands=1, 
            eType=gdal.GDT_Float32)
    
        dataset.SetGeoTransform((
            x_min,    # 0
            PIXEL_SIZE,  # 1
            0,  # 2
            y_max,    # 3
            0,                      # 4
            -PIXEL_SIZE))  
    
        dataset.SetProjection(dest_crs.ExportToWkt())
        array = self.array
#        array[array==self.header['NODATA_value']]=np.nan
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.GetRasterBand(1).SetNoDataValue(self.header['NODATA_value'])
        if filename is not None:
            dataset.FlushCache()  # Write to disk.
            dataset = None
            return dst_filename
        else:
            return dataset

#%%=========================== private functions ==============================
    def __osgeoDS2raster(self, ds):
        """
        convert an osgeo dataset to a raster object
        """
        array = ds.ReadAsArray()
        geoT = ds.GetGeoTransform()
        projection = ds.GetProjection()
        left = geoT[0]
        top = geoT[3]
        cellsize = geoT[1]
        nrows = ds.RasterYSize
        ncols = ds.RasterXSize
        xllcorner = left
        yllcorner = top - cellsize*nrows
        NODATA_value = ds.GetRasterBand(1).GetNoDataValue()
        if NODATA_value is None:
            NODATA_value = -9999
        header = {'ncols':ncols, 'nrows':nrows,
                  'xllcorner':xllcorner, 'yllcorner':yllcorner,                  
                  'cellsize':cellsize, 'NODATA_value':NODATA_value}
        newObj = Raster(array=array, header=header, projection=projection)
        return newObj
        
    def __header2extent(self):
        """
        To convert header (dict) to a spatial extent of the DEM
        extent: (left, right, bottom, top)
        """
        R = self.header
        left = R['xllcorner']
        right = R['xllcorner']+R['ncols']*R['cellsize']
        bottom = R['yllcorner']
        top = R['yllcorner']+R['nrows']*R['cellsize']
        self.extent = (left, right, bottom, top)
        self.extent_dict = {'left':left, 'right':right, 'bottom':bottom, 'top':top}

    def __map2sub(self, X, Y):
        """
        convert map points to subscripts of a matrix with geo reference header
        X, Y: coordinates in map units
        return
            rows, cols: (int) subscripts of the data matrix
        """
        #x and y coordinate of the centre of the first cell in the matrix
        X = np.array(X)
        Y = np.array(Y)
        header = self.header
        
        x0 = header['xllcorner']+0.5*header['cellsize']
        y0 = header['yllcorner']+(header['nrows']-0.5)*header['cellsize']
        rows = (y0-Y)/header['cellsize'] # row and col number starts from 0
        cols = (X-x0)/header['cellsize']
        if isinstance(rows, np.ndarray):
            rows = rows.astype('int64')
            cols = cols.astype('int64') #.astype('int64')
        else:
            rows = int(rows)
            cols = int(cols)
        return rows, cols

    def __sub2map(self, rows, cols):
        """
        convert subscripts of a matrix to map coordinates 
        rows, cols: subscripts of the data matrix, starting from 0
        return
            X, Y: coordinates in map units
        """
        #x and y coordinate of the centre of the first cell in the matrix
        if not isinstance(rows, np.ndarray):
            rows = np.array(rows)
            cols = np.array(cols)        
        
        header = self.header
        left = self.extent[0] #(left, right, bottom, top)
        top = self.extent[3]
        X = left + (cols+0.5)*header['cellsize']
        Y = top  - (rows+0.5)*header['cellsize']
         
        return X, Y

# read ascii file        
    def __read_asc(self, num_header_rows=6):
        """
        read asc file and return array, header
        if self.source_file ends with '.gz', then read the compressed file
        """
        fileName = self.source_file
        try:
            fh = open(fileName, 'r')
            fh.close()
        # Store configuration file values
        except FileNotFoundError:
            # Keep preset values
            print('Error: '+fileName+' does not appear to exist')
            return
        # read header
        header = {} # store header information including ncols, nrows, ...        
        n=1
        if fileName.endswith('.gz'):
            # read header
            with gzip.open(fileName, 'rt') as f:                
                for line in f:
                    if n<=num_header_rows:
                        line = line.split(" ", 1)
                        header[line[0]] = float(line[1])
                    else:
                        break
                    n = n+1
        else:
            # read header
            with open(fileName, 'rt') as f:            
                for line in f:
                    if n<=num_header_rows:
                        line = line.split(" ", 1)
                        header[line[0]] = float(line[1])
                    else:
                        break
                    n = n+1
    # read value array
        array  = np.loadtxt(fileName, skiprows=num_header_rows, dtype='float64')
        if 'NODATA_value' not in header.keys():
            header['NODATA_value'] = -9999
        array[array == header['NODATA_value']] = float('nan')
        header['ncols']=int(header['ncols'])
        header['nrows']=int(header['nrows'])
        self.array = array
        self.header = header
        prjFile = self.source_file[:-4]+'.prj'
        if os.path.isfile(prjFile):
            with open(prjFile, 'r') as file:
                projection = file.read()
            self.projection = projection
        return None

# read GTiff file
    def __read_tif(self):
        """
        read tif file and return array, header
        only read the first band
        """
        from osgeo import gdal
        tifName = self.source_file
        ds = gdal.Open(tifName)        
        ncols = ds.RasterXSize
        nrows = ds.RasterYSize
        geoTransform = ds.GetGeoTransform()
        x_min = geoTransform[0]
        cellsize = geoTransform[1]
        y_max = geoTransform[3]
        xllcorner = x_min
        yllcorner = y_max - nrows*cellsize
        rasterBand = ds.GetRasterBand(1)
        NODATA_value = rasterBand.GetNoDataValue()
        array = rasterBand.ReadAsArray()
        header = {'ncols':ncols, 'nrows':nrows, 
                  'xllcorner':xllcorner, 'yllcorner':yllcorner, 
                  'cellsize':cellsize, 'NODATA_value':NODATA_value}        
        if not np.isscalar(header['NODATA_value']):
            header['NODATA_value'] = -9999
        array[array == header['NODATA_value']] = float('nan')
        self.header = header
        self.array = array
        self.projection = ds.GetProjection()
        rasterBand = None
        ds = None
        return None

    def __read_bytes(self):
        """ Read file from a bytes object
        """
        f = self.source_file
        # read header
        header = {} # store header information including ncols, nrows, ...
        num_header_rows = 6
        for _ in range(num_header_rows):
            line = f.readline()
            line = line.strip().decode("utf-8").split(" ", 1)
            header[line[0]] = float(line[1])
            # read value array
        array  = np.loadtxt(f, skiprows=num_header_rows, dtype='float64')
        array[array == header['NODATA_value']] = float('nan')
        header['ncols'] = int(header['ncols'])
        header['nrows'] = int(header['nrows'])
        self.array = array
        self.header = header

    def __SetWktProjection(self, epsg_code):
        """
        get coordinate reference system (crs) as Well Known Text (WKT) 
            from https://epsg.io
        epsg_code: the epsg code of a crs, e.g. BNG:27700, WGS84:4326
        return wkt text
        """
        import requests
        # access projection information
        wkt = requests.get('https://epsg.io/{0}.prettywkt/'.format(epsg_code))
        # remove spaces between charachters
        remove_spaces = wkt.text.replace(" ", "")
        # place all the text on one line
        output = remove_spaces.replace("\n", "")
        self.projection = output
        return output
#%% Extent compare between two Raster objects
def compare_extent(extent0, extent1):
    """Compare and show the difference between two Raster extents
    extent0, extent1: objects or extent dicts to be compared
    displaye: whether to show the extent in figures
    Return:
        0 extent0>=extent1
        1 extent0<extent1
        2 extent0 and extent1 have intersections
    """
    logic_left = extent0[0]<=extent1[0]
    logic_right = extent0[1]>=extent1[1]
    logic_bottom = extent0[2]<=extent1[2]
    logic_top = extent0[3]>=extent1[3]
    logic_all = logic_left+logic_right+logic_bottom+logic_top
    if logic_all == 4:
        output = 0
    elif logic_all == 0:
        output = 1
    else:
        output = 2
        print(extent0)
        print(extent1)
    return output

#%% Combine raster files
def combine_raster(asc_files, num_header_rows=6):
    """Combine a list of asc files to a DEM Raster
    asc_files: a list of asc file names
    all raster files have the same cellsize
    """
    # default values for the combined Raster file
    xllcorner_all = []
    yllcorner_all = []
    extent_all =[]
    
    # read header
    for file in asc_files:
        header0 = _read_header(file, num_header_rows)
        extent0 = header2extent(header0)
        xllcorner_all.append(header0['xllcorner'])
        yllcorner_all.append(header0['yllcorner'])
        extent_all.append(extent0)
    cellsize = header0['cellsize']
    if 'NODATA_value' in header0.keys():
        NODATA_value = header0['NODATA_value']
    else:
        NODATA_value = -9999
    xllcorner_all = np.array(xllcorner_all)
    xllcorner = xllcorner_all.min()
    yllcorner_all = np.array(yllcorner_all)
    yllcorner = yllcorner_all.min()
    extent_all = np.array(extent_all)
    x_min = np.min(extent_all[:,0])
    x_max = np.max(extent_all[:,1])
    y_min = np.min(extent_all[:,2])
    y_max = np.max(extent_all[:,3])
#    extent = (x_min, x_max, y_min, y_max)
#    print(extent)
    nrows = int((y_max-y_min)/cellsize)
    ncols = int((x_max-x_min)/cellsize)
    header = header0.copy()
    header['xllcorner'] = xllcorner
    header['yllcorner'] = yllcorner
    header['ncols'] = ncols
    header['nrows'] = nrows
    header['NODATA_value'] = NODATA_value
    array = np.zeros((nrows ,ncols))+NODATA_value
    print(array.shape)
    for file in asc_files:
        obj0 = Raster(file, num_header_rows=num_header_rows)
        x0 = obj0.extent[0]+obj0.header['cellsize']/2
        y0 = obj0.extent[3]-obj0.header['cellsize']/2
        row0, col0 = _map2sub(x0, y0, header)
        array[row0:row0+obj0.header['nrows'],
              col0:col0+obj0.header['ncols']] = obj0.array
    array[array == header['NODATA_value']] = float('nan')
    obj_output = Raster(array=array, header=header)
    return obj_output
   
#%% shapePoints= makeDiagonalShape(extent)
def makeDiagonalShape(extent):
    #extent = (left, right, bottom, top)
    shapePoints = np.array([[extent[0], extent[2]], 
                           [extent[1], extent[2]], 
                           [extent[1], extent[3]], 
                           [extent[0], extent[3]]])
    return shapePoints

#%% convert header data to extent
def header2extent(demHead):
    # convert dem header file (dict) to a spatial extent of the DEM
    R = demHead
    left = R['xllcorner']
    right = R['xllcorner']+R['ncols']*R['cellsize']
    bottom = R['yllcorner']
    top = R['yllcorner']+R['nrows']*R['cellsize']
    extent = (left, right, bottom, top)
    return extent         

def _read_header(file_name, num_header_rows=6):
    """ Read and return a header dict from a asc file
    read the header of an asc file and return header
    if file_name ends with '.gz', then read the compressed file
    """
    # read header
    header = {} # store header information including ncols, nrows,...
    n=1
    if file_name.endswith('.gz'):
        # read header
        with gzip.open(file_name, 'rt') as f:                
            for line in f:
                if n<=num_header_rows:
                    line = line.split(" ",1)
                    header[line[0]] = float(line[1])
                else:
                    break
                n = n+1
    else:
        # read header
        with open(file_name, 'rt') as f:            
            for line in f:
                if n<=num_header_rows:
                    line = line.split(" ",1)
                    header[line[0]] = float(line[1])
                else:
                    break
                n = n+1
    return header

#%% rows, cols = _map2sub(X, Y, header)
def _map2sub(X, Y, header):
    """ convert map coordinates to subscripts of an array
    array is defined by a geo-reference header
    X, Y: a scalar or numpy array of coordinate values
    Return: rows, cols in the array
    """
    # X and Y coordinate of the centre of the first cell in the array
    x0 = header['xllcorner']+0.5*header['cellsize']
    y0 = header['yllcorner']+(header['nrows']-0.5)*header['cellsize']
    rows = (y0-Y)/header['cellsize'] # row and col number starts from 0
    cols = (X-x0)/header['cellsize']
    if isinstance(rows, np.ndarray):
        rows = rows.astype('int64')
        cols = cols.astype('int64') #.astype('int64')
    else:
        rows = int(rows)
        cols = int(cols)
    return rows, cols
   
#%%
def merge(obj_origin, obj_target, resample_method='bilinear'):
    """Merge the obj_origin to obj_target
    assign grid values in the origin Raster to the cooresponding grid cells in
    the target object. If cellsize are not equal, the origin Raster will be
    firstly resampled to the target object.
    obj_origin, obj_target: Raster objects
    """
    if obj_origin.header['cellsize'] != obj_target.header['cellsize']:
        obj_origin = obj_origin.resample(obj_target.header['cellsize'], 
                                   method=resample_method)
#    else:
#        obj_origin = self
    grid_x, grid_y = obj_origin.GetXYcoordinate()
    rows, cols = _map2sub(grid_x, grid_y, obj_target.header)
    ind_r = np.logical_and(rows >= 0, rows <= obj_target.header['nrows']-1)
    ind_c = np.logical_and(cols >= 0, cols <= obj_target.header['ncols']-1)
    ind = np.logical_and(ind_r, ind_c)
    ind = np.logical_and(ind, ~np.isnan(obj_origin.array))
    obj_output = copy.deepcopy(obj_target)
    obj_output.array[rows[ind], cols[ind]] = obj_origin.array[ind]
    return obj_output

def main():
    print('Package to deal with raster data')

if __name__=='__main__':
    main()
    
    

