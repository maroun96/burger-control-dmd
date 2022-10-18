import argparse
import h5py
from pathlib import Path

from utils import merge_data, load_yml, model_hdf5
from dmd import DMDc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dpath", type=str, default='datasets',
                    help="path of directory that contains the dataset")
    parser.add_argument("-mp", "--mpath", type=str, default='model',
                    help="path of directory in which the model is saved")


    args = parser.parse_args()
    dataset_dirname = args.dpath
    model_dirname = args.mpath


    X, Y, C = merge_data(Path(dataset_dirname) / 'main.hdf5')

    cfg = load_yml('config.yml')
    params = cfg["dmd_params"]
    p = params["augstate_trunc"]
    r = params["output_trunc"]

    Atilde, Btilde, Ur = DMDc(X.T, Y.T, C.T, p, r)

    model_hdf5(dir_name=model_dirname, attr=params, A=Atilde, B=Btilde, Ur=Ur)
    