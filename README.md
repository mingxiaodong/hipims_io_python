hipims_io
--------
To install hipims_io
```
pip install hipims_io
```
A demonstration to setup an HiPIMS input object with a sample DEM:
>>> import hipims_io as hp
>>> obj_in = hp.demo_hipims_input() # create an input object and show

To setup an input object for HiPIMS, you will need at least DEM file and do:
```
>>> from hipims_io import InputHipims
>>> obj_in = InputHipims(dem_data='your_file.asc') # create object
>>> obj_in.domain_show() # show domain map
>>> print(obj_in) # print model summary
>>> obj_in.write_input_files() # write all input files for HiPIMS
``` 
