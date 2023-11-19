Tif_split_merge - readme
***************************************************

This program is for splitting tif to color channels - and saves as tif files (with Z stacks).
Also it can merge channels files into one tif file, 

Basic instruction:

split:
1. choose .tif file
2. give names to the channels (GFP, DAPI...), seperated by spaces, line or commas - then click OK
3. copy the Matlab command (make sure to add X:\Amichay\Resources to path)
4. choose all channels, choose which Z stacks to use (leave level at 10)

merge:
1. choose .tif files to merge from the "log_output" freshly created folder - for probes use the X_STACK_ADDED.tif, for DAPI use X_STACK_ORIG
2. chose the order of channels on the comboboxes, press "merge"
3. choose filename and save
The ouput will be both the merged tif and also the max projection.

Roy
