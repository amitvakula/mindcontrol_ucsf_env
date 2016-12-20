#!/usr/bin/env python
import pymongo
import nibabel as nib
from pbr.config import config as cc
from os.path import join, exists, split
from dipy.tracking import utils
import argparse
from nipype.utils.filemanip import load_json
import os
import numpy as np
from dipy.tracking.utils import subsegment

def get_collection(port=3001):
    from pymongo import MongoClient
    client = MongoClient("localhost", port)
    db =  client.meteor
    collection = db.subjects
    return collection, client
    
def get_papaya_aff(img):
    vs = img.header.get_zooms()
    aff = img.get_affine()
    ort = nib.orientations.io_orientation(aff)
    papaya_aff = np.zeros((4, 4))
    for i, line in enumerate(ort):
        papaya_aff[line[0],i] = vs[i]*line[1]
    papaya_aff[:, 3] = aff[:, 3]
    return papaya_aff
    

def convert_to_volume(drawing, papaya_aff, aff, img, do_subsegment = True):

    topoints = lambda x : np.array([[m["x"], m["y"], m["z"]] for m in x["world_coor"]])
    points_orig = list(map(topoints, drawing))
    if do_subsegment:
        points = list(subsegment(points_orig, 0.5))
    else:
        points = points_orig
    mask2 = utils.density_map(points, img.shape, affine=papaya_aff)
    points_nifti_space = list(utils.move_streamlines(points, aff, input_space=papaya_aff))
    mask1 = utils.density_map(points_nifti_space, img.shape, affine=aff)

    #print((mask1 == mask2).all())
    # img1 = nib.Nifti1Image(mask1)
    #print(mask1.sum(), mask2.sum())
    
    return mask1, points_nifti_space

def convert_to_indices(streamline, papaya_aff, aff, img):
    topoints = lambda x : np.array([[m["x"], m["y"], m["z"]] for m in x["world_coor"]])
    points_orig = topoints(streamline)
    points_nifti_space = list(utils.move_streamlines([points_orig], aff, input_space=papaya_aff))[0]
    from dipy.tracking._utils import _to_voxel_coordinates, _mapping_to_voxel
    lin_T, offset = _mapping_to_voxel(aff, None)
    idx = _to_voxel_coordinates(streamline, lin_T, offset)
    return points_nifti_space, idx

def create_volume(output):
    from pbr.config import config as cc
    img = nib.load(join(cc["output_directory"],output["check_masks"][0]))
    aff = img.get_affine() #affine()
    papaya_affine = get_papaya_aff(img)
    data = np.zeros(img.shape)
    outputfiles = []
    mse = output["subject_id"]
    sequence = output["name"]
    for contour in output["contours"]:
        drawing = contour["contours"]
        name = contour['name'].replace(" ", "_")
        author = contour["checkedBy"]
    
        mask, points_nifti_space = convert_to_volume(drawing, papaya_affine, aff, img)
        outfilepath = join(mse, "mindcontrol/{}/{}/rois".format(sequence, output["entry_type"]))
        if not exists(join(cc["output_directory"], outfilepath)):
            os.makedirs(join(cc["output_directory"],outfilepath))
            print(join(cc["output_directory"], outfilepath), "created")
        outfilename = join(outfilepath, "{}-{}.nii.gz".format(name,author))
        nib.Nifti1Image(mask.astype(np.float32), affine=aff).to_filename(join(cc["output_directory"], outfilename))
        
        print("wrote", join(cc["output_directory"], outfilename))
        outputfiles.append(outfilename)

        if (papaya_affine == aff).all():
            print("affines are the same")
        else:
            print(papaya_affine - aff)

        mask, points_nifti_space = convert_to_volume(drawing, aff, aff, img)
        outfilename2 = outfilename.replace(".nii.gz", "_origAff.nii.gz")
        nib.Nifti1Image(mask.astype(np.float32), affine=aff).to_filename(join(cc["output_directory"], outfilename2))
        print("wrote", join(cc["output_directory"], outfilename2))
        outputfiles.append(outfilename2)

    return outputfiles

def paintVolume(drawing, papaya_affine, aff, img, outfilepath, name, authors, suffix=""):
    import pandas as pd
    df = pd.DataFrame()

    for i,d in enumerate(drawing):
        pv = d["paintValue"]
        mask_tmp, points_nii_space = convert_to_volume([d], papaya_affine, aff, img, False)
        nii_points = points_nii_space[0]
        tmp = []
        for ni in nii_points:
            tmp.append({"x": ni[0], "y":ni[1], "z": ni[2], "val": pv})
        df = df.append(pd.DataFrame(tmp), ignore_index=True)
    df.drop_duplicates(inplace=True)
    if not exists(join(cc["output_directory"], outfilepath)):
        os.makedirs(join(cc["output_directory"],outfilepath))
        print(join(cc["output_directory"], outfilepath), "created")
    outfilename = join(outfilepath, "{}-{}{}.csv".format(name,"-".join(authors), suffix))
    #nib.Nifti1Image(mask.astype(np.float32), affine=aff).to_filename(join(cc["output_directory"], outfilename))
    df.to_csv(join(cc["output_directory"], outfilename))
    return outfilename

def create_paint_volume(output):
    from pbr.config import config as cc
    img = nib.load(join(cc["output_directory"],output["check_masks"][-1]))
    aff = img.get_affine() #affine()
    papaya_affine = get_papaya_aff(img)
    data = np.zeros(img.shape)
    outputfiles = []
    mse = output["subject_id"]
    sequence = output["name"]
    #for c in output["contours"]:
    drawing_old = output["painters"]
    drawing = []
    for p in drawing_old:
        if len(p["world_coor"]):
            drawing.append(p)

    name = output["entry_type"] #contour['name'].replace(" ", "_")
    authors = set([q["checkedBy"] for q in drawing])

    #mask, points_nifti_space = convert_to_volume(drawing, papaya_affine, aff, img, False)

    outfilepath = join(mse, "mindcontrol/{}/{}/rois".format(sequence, output["entry_type"]))

    outfilename = paintVolume(drawing, papaya_affine, aff, img, outfilepath, name, authors)
    print("painter wrote", join(cc["output_directory"], outfilename))
    outputfiles.append(outfilename)

    outfilename = paintVolume(drawing, aff, aff, img, outfilepath, name, authors, suffix="_origAff")
    print("painter wrote", join(cc["output_directory"], outfilename))
    outputfiles.append(outfilename)
    return outputfiles


def convert_to_real_world(drawing, papaya_aff, aff, img):

    topoints = lambda x : np.array([[m["world_coor"]["x"],
                                     m["world_coor"]["y"],
                                     m["world_coor"]["z"]] for m in x])
    points_orig = topoints(drawing)
    #print(points_orig)
    points_nifti_space = list(utils.move_streamlines(points_orig, aff, input_space=papaya_aff))
    mask1 = utils.density_map(points_nifti_space, img.shape, affine=aff)
    return mask1, points_nifti_space

def create_points_df(entry, points_nii_space, suffix=""):
    import pandas as pd
    df = pd.DataFrame(points_nii_space, columns=["x","y","z"])
    annot = []
    authors = []
    for e in entry["loggedPoints"]:
        authors.append(e["checkedBy"])
        if "note" in entry.keys():
            annot.append(e["note"])
        else:
            annot.append(None)
    df["annotation"] = annot
    df["author"] = authors
    mse = entry["subject_id"]
    sequence = entry["name"]
    outfilepath = join(mse, "mindcontrol/{}/{}/rois".format(sequence, entry["entry_type"]))
    if not exists(join(cc["output_directory"], outfilepath)):
        os.makedirs(join(cc["output_directory"],outfilepath))
        print(join(cc["output_directory"], outfilepath), "created")
    author = "-".join(set(authors))
    outfilename = join(outfilepath, "{}-{}{}.csv".format(entry["name"], author, suffix))
    df.to_csv(join(cc["output_directory"], outfilename))
    return join(cc["output_directory"], outfilename)

def get_all_seeds(mse, meteor_port, entry_types=None):
    import pandas as pd
    coll, cli = get_collection(meteor_port+1)
    finder = {"subject_id": mse}
    if entry_types is not None:
        finder["entry_type"] = {"$in": entry_types}
    entries = coll.find(finder)
    saved = []
    for entry in entries:
        if "name" in entry.keys(): #i.e. there needs to be a sequence associated w/ the ROI
            name = entry["name"]
            et = entry["entry_type"]
            sid = entry["subject_id"]
            if "loggedPoints" in entry.keys():
                img = nib.load(join(cc["output_directory"],entry["check_masks"][0]))
                aff = img.get_affine() #affine()
                papaya_affine = get_papaya_aff(img)
                mask1, points_nii_space = convert_to_real_world(entry["loggedPoints"], papaya_affine,
                                                                aff, img)
                outfilename = create_points_df(entry, points_nii_space)
                print("wrote", outfilename)
                mask2, points_nii_space2 = convert_to_real_world(entry["loggedPoints"], aff,
                                                                aff, img)
                outfilename2 = create_points_df(entry, points_nii_space2)
                print("write", outfilename2)

def get_all_contours(mse, meteor_port, entry_types = None):
    coll, cli = get_collection(meteor_port+1)
    finder = {"subject_id": mse}
    if entry_types is not None:
        finder["entry_type"] = {"$in": entry_types}
    entries = coll.find(finder)
    saved = []
    for entry in entries: 
        if "name" in entry.keys(): #i.e. there needs to be a sequence associated w/ the ROI
            name = entry["name"]
            et = entry["entry_type"]
            sid = entry["subject_id"]
            if "contours" in entry.keys():
                print("found drawings for", name, et, sid)
                outputs = create_volume(entry)
                if outputs:
                    for o in outputs:
                        if not o in entry["check_masks"]:
                            entry["check_masks"].append(o)
                    coll.update_one({"name": name, "subject_id":sid, "entry_type": et},
                                    {"$set":{"check_masks": entry["check_masks"]}})
            

def get_all_paints(mse, meteor_port, entry_types = None):
    coll, cli = get_collection(meteor_port+1)
    finder = {"subject_id": mse}
    if entry_types is not None:
        finder["entry_type"] = {"$in": entry_types}
    entries = coll.find(finder)
    saved = []
    for entry in entries:
        if "name" in entry.keys(): #i.e. there needs to be a sequence associated w/ the ROI
            name = entry["name"]
            et = entry["entry_type"]
            sid = entry["subject_id"]
            if "painters" in entry.keys():
                print("found paintings for", name, et, sid)
                outputs = create_paint_volume(entry)

if __name__ == '__main__':
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

    
