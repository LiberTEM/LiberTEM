
[Feature] Added support for seq files to load excluded pixels from xml (:pr:`805`, :pr:`1077`)
====
    - added function xml_map_sizes(mad pixel map)
    - added function xml_unbinned_map_maker(xy_map_sizes)
    - added function xml_map_index_selector(unbinned_x, unbinned_y, map_sizes)
    - added function xml_defect_coord_extractor(bad_pixel_map, map_index, map_sizes)
    - added function xml_defect_data_extractor(root)
    - added function bin_array_2d(a,binning)
    - added function array_cropping(arr, start_size, req_size, offset)
    - added function xml_generate_map_size(exc_rows, exc_cols, exc_pix, size, metadata)
    - added function xml_processing(tree, metadata_dict)
    - added function load_xml_from_string(xml, metadata)
    - added function _load_xml_from_file(self, path)
    - declared excluded pixels in __init__ of seq.py
    - defined the value of self._excluded_pixels in maybe_load_dark_gain(self) function
    - the get_correction_data(self) function now also returns the self._excluded_pixels
