hipims_io
--------
To install hipims_io
```
pip install hipims_io
```
To setup an input object for HiPIMS, you will need at least a DEM file, and simply do::

    >>> import hipims_io as hp
    >>> obj_in = hp.demo_hipims_input() # create an input object and show
    >>> obj_in.write_input_files() # write all input files for HiPIMS
    
