#!/usr/bin/env python

# dependencies
import pymongo
import nibabel as nib
from os.path import join, exists, split
from dipy.tracking import utils
import os
import numpy as np
from dipy.tracking.utils import subsegment

def get_collection(port=3001):
    from pymongo import MongoClient
    client = MongoClient("localhost", port)
    db =  client.meteor
    collection = db.subjects
    return collection, client

def get_segmentation_mask(db, subject_id="mse65"):
    cursor = db.find({"subject_id":subject_id, "entry_type":"freesurfer"})
    results = []
    for item in cursor:
        results.append(item)
    assert len(results) == 1, "more than one result! either modify the code or restrict your query Amit"
    subject = results[0]
    DATA_HOME = "/data/henry7/PBR/subjects/"
    subject["check_masks"][1]
    return nib.load(DATA_HOME + subject["check_masks"][1])

def get_papaya_aff(img):
    vs = img.header.get_zooms()
    aff = img.get_affine()
    ort = nib.orientations.io_orientation(aff)
    papaya_aff = np.zeros((4, 4))
    for i, line in enumerate(ort):
        papaya_aff[line[0],i] = vs[i]*line[1]
    papaya_aff[:, 3] = aff[:, 3]
    return papaya_aff

def convert_to_indices(streamline, papaya_aff, aff, img):
    #print(streamline)
    topoints = lambda x : np.array([[m["x"], m["y"], m["z"]] for m in x["world_coor"]])
    points_orig = topoints(streamline)
    points_nifti_space = list(utils.move_streamlines([points_orig], aff, input_space=papaya_aff))[0]
    from dipy.tracking._utils import _to_voxel_coordinates, _mapping_to_voxel
    lin_T, offset = _mapping_to_voxel(aff, None)
    idx = _to_voxel_coordinates(points_orig, lin_T, offset)
    return points_nifti_space, idx

# the actual paint function (put helper functions above where I convert to the right space?)
def get_points_to_paint(drawing, papaya_affine, aff, img, outfilepath, name, authors, suffix=""):
    import pandas as pd
    df = pd.DataFrame()

    for d in drawing:
        pv = d["paintValue"]
        points_nii_space, trans_points = convert_to_indices(d, papaya_affine, aff, img)
        tmp = []
        for ni in trans_points:
            tmp.append({"x": ni[0], "y":ni[1], "z": ni[2], "val": pv})
        df = df.append(pd.DataFrame(tmp), ignore_index=True)
    df.drop_duplicates(inplace=True)
    #if not exists(join(cc["output_directory"], outfilepath)):
        #os.makedirs(join(cc["output_directory"],outfilepath))
        #print(join(cc["output_directory"], outfilepath), "created")
    #outfilename = join(outfilepath, "{}-{}{}.csv".format(name,"-".join(authors), suffix))
    #nib.Nifti1Image(mask.astype(np.float32), affine=aff).to_filename(join(cc["output_directory"], outfilename))
    #df.to_csv(join(cc["output_directory"], outfilename))
    #return outfilename
    return df

def create_paint_volume(drawing, img, output, outfilepath):
    #from pbr.config import config as cc
    #img = nib.load(join(cc["output_directory"],output["check_masks"][-1]))
    aff = img.get_affine() #affine()
    papaya_affine = get_papaya_aff(img)
    data = np.zeros(img.shape)
    outputfiles = []
    mse = output["subject_id"]
    sequence = output["name"]
    #for c in output["contours"]:
    drawing_old = output["painters"]
    #drawing = []
    #for p in drawing_old:
    #    if len(p["world_coor"]):
    #        drawing.append(p)

    name = output["entry_type"] #contour['name'].replace(" ", "_")
    authors = set([q["checkedBy"] for q in drawing])

    data = img.get_data() + data
    
    df = get_points_to_paint(drawing, papaya_affine, aff, img, outfilepath, name, authors, suffix="")
    for item in df.iterrows():
        paint_value = item[1]['val']
        x,y,z = item[1]['x'], item[1]['y'], item[1]['z']
        data[x][y][z] = paint_value
    
    painted_image = nib.nifti1.Nifti1Image(data,aff,img.header)
    
    nib.save(painted_image,os.path.join(outfilepath,"painted.nii.gz"))
    
        
    #mask, points_nifti_space = convert_to_volume(drawing, papaya_affine, aff, img, False)

    #outfilepath = join(mse, "mindcontrol/{}/{}/rois".format(sequence, output["entry_type"]))

    #papaya_aff_points = paintVolume(drawing, papaya_affine, aff, img, outfilepath, name, authors)
    #print("painter wrote", join(cc["output_directory"], outfilename))
    #outputfiles.append(outfilename)

    #aff_points = paintVolume(drawing, aff, aff, img, outfilepath, name, authors, suffix="_origAff")
    #print("painter wrote", join(cc["output_directory"], outfilename))
    #outputfiles.append(outfilename)
    
    #to_rtn = paint_over()
    
    return painted_image


if __name__ == 'IGNORE':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', dest="env")
    parser.add_argument("-s", dest="subjects", nargs="+")
    parser.add_argument("--entry_type", dest="entry_types", nargs="+")
    config = load_json(os.path.join(os.path.split(__file__)[0], "config.json"))
    args = parser.parse_args()
    print(args)
    if args.env in ["development", "production"]:
        env = args.env
        if len(args.subjects) > 0:
            if args.subjects[0].endswith(".txt"):
                import numpy as np
                subjects = np.genfromtxt(args.subjects[0], dtype=str)
            else:
                subjects = args.subjects
        for mse in subjects:
            meteor_port = config[env]["meteor_port"]
            try:
                get_all_contours(mse, meteor_port, args.entry_types)
            except IndexError:
                print("ANISHA NEEDS TO FIX")

            get_all_seeds(mse, meteor_port, args.entry_types)
            try:
                get_all_paints(mse, meteor_port, args.entry_types)
            except ValueError:
                print("ERROR WITH PAINTING", mse)
    else:
        raise Exception("Choose the database you want to append to w/ -e production or -e development")

if __name__ == "__main__":
    collection, client = get_collection(5051)
    seg_mask = get_segmentation_mask(collection, subject_id="mse65")
    
    # getting painter object
    cursor = collection.find({"subject_id":"mse65", "entry_type":"lst"})
    results = []
    for item in cursor:
        results.append(item)
    assert len(results) == 1, "more than one result! either modify the code or restrict your query Amit"
    painter_entry = results[0]
    modified_segmentation = create_paint_volume([painter_entry['painters'][0]], seg_mask, painter_entry, "/Users/amitvakula/Documents/research/results")



    diff_map = seg_mask.get_data() - modified_segmentation.get_data()
    for i in range(np.shape(diff_map)[0]):
        for j in range(np.shape(diff_map)[1]):
            for k in range(np.shape(diff_map)[2]):
                if diff_map[i][j][k] != 0:
                    print("at {} we got value {}".format((i,j,k),diff_map[i][j][k]))
