import os
import h5py

root = "./data/star20_kh1_100_15_n100_80000_noise0/"

bad = []
good = 0

for dirpath, _, filenames in os.walk(root):
    for name in filenames:
        path = os.path.join(dirpath, name)

        # Optional: only check likely data files
        # if not (name.endswith(".mat") or name.endswith(".h5") or name.endswith(".hdf5")):
        #     continue

        try:
            with h5py.File(path, "r") as f:
                pass
            good += 1
        except Exception as e:
            bad.append((path, repr(e)))
print('good\n')
print(good)
print('bad\n')
print(len(bad))

for path, err in bad[:50]:
    print(path)
    print("   ", err)
