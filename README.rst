Hipims_io
--------

To setup an input object for HiPIMS, you will need at least a DEM file, and simply do::

    >>> from hipims_io import InputHipims
    >>> obj_input = InputHipims(dem_file) # create an input object
    >>> obj_input.write_input_files() # write all input files for HiPIMS