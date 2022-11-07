import argparse
import h5py
from pathlib import Path

from utils import H5pyHandler, load_yml, model_hdf5
from dmd import DMD,DMDc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=True, 
                    action=argparse.BooleanOptionalAction)


    args = parser.parse_args()
    control_bool = args.control
    iohandler = H5pyHandler(with_control=control_bool)


    X = iohandler.merge_data(group_name="input_arrays")
    Y = iohandler.merge_data(group_name="output_arrays")
    if control_bool:
        C = iohandler.merge_data(group_name="control_arrays")

    cfg = load_yml('config.yml')
    params = cfg["dmd_params"]
    if control_bool:
        p = params["augstate_trunc"]
    r = params["output_trunc"]

    if control_bool:
        Atilde, Btilde, Ur = DMDc(X.T, Y.T, C.T, p, r)
        iohandler.model_hdf5(attr=params, A=Atilde, Ur=Ur, B=Btilde)
    else:
        Atilde, Ur = DMD(X.T, Y.T, r)
        iohandler.model_hdf5(attr=params, A=Atilde, Ur=Ur)

    

    print("DMD Done !")
    