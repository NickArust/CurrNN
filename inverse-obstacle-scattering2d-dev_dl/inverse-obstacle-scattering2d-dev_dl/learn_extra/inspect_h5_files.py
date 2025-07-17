import h5py

def inspect_mat_file(file_name):
    """
    Inspect a .mat file (MATLAB v7.3 format) and print details about its contents,
    including structure and properties of all groups and datasets.
    """
    with h5py.File(file_name, 'r') as file:
        def print_details(name, obj):
            print(f"Name: {name}")
            if isinstance(obj, h5py.Dataset):
                print("  Type: Dataset")
                print("  Shape:", obj.shape)
                print("  Data Type:", obj.dtype)
                if obj.dtype == 'object':  # Check for references, common in MATLAB structures
                    try:
                        # Attempt to dereference objects if possible
                        ref_data = obj[()]
                        print("  Reference Data:", ref_data)
                        for ref in np.nditer(ref_data, flags=['refs_ok']):
                            if ref:
                                dereferenced = file[ref]
                                print("    Dereferenced Data:", dereferenced[:])
                    except Exception as e:
                        print("  Could not dereference:", e)
            else:
                print("  Type: Group")
        file.visititems(print_details)

# Replace 'yourmatfile.mat' with the path to your MATLAB v7.3 .mat file.
filename = "/lustre/fs1/home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn/data/star5_kh10_n48_500/valid_data.mat"
inspect_mat_file(filename)

