# Scout Dataset

TODO:
Dataset class for pytorch dataset. Uses individual as base data format for fast loading
Geometry class with projection, mesh loading etc
Example notebook showing usage of various functions. Loading, Visualisation, Projection etc
Readmes with usage information


Handle camera renames in accordance with: {f'cvlabrpi{i}':f'cam{new}' for new, i in enumerate([10,21,13,12,7,19,24,5,23,3,2,4,1,22,11,8,26,17,14,25,6,9,18,15,16])}
Follow planned directory structure:
scout:
scout toolkit:
scout/scouttoolkit
scout/dataset/images/sequence_01/cameraname/image_i.jpg
scout/dataset/annotations/sequence_01/mot/camera_name.txt
scout/dataset/annotations/sequence_01/individual/camera_name/frameid.txt
scout/dataset/annotations/sequence_01/coco/annotations.json
scout/dataset/calibrations/sequence_01.json
scout/dataset/calibrations/sequence_02.json
scout/dataset/timestamps/sequence_01.json
scout/dataset/timestamps/sequence_02.json
scout/dataset/mesh/highres.ply
scout/dataset/mesh/lowres.ply