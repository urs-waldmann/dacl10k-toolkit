from dacl10k.dacl10kdataset import Dacl10kDataset

split = "validation"  # "train", "validation", "testdev" (later also "testchallenge")
path_to_data = "/home/urs/Documents/data/dacl10k_v2_devphase"
dataset = Dacl10kDataset(split, path_to_data, resize_mask=(512, 512), resize_img=(512, 512), normalize_img=True)

dataset.run_prefetching(n_jobs=12)

print(len(dataset.prefetched_data), type(dataset.prefetched_data))

path_to_store_prefechted_data = "/home/urs/Documents/data/dacl10k_v2_devphase/prefetched/"
# you can also pass a filename if you want otherwise the split name ("validation.pkl") is used.
dataset.save_prefetched_data(path_to_store_prefechted_data)
del dataset
