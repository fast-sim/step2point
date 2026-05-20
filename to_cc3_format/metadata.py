import os

import numpy as np


class Metadata:
    """
    By default, cooredinates are mm, and positions are in
    the physical detector space.

    Things labeled 'raw' refer to the data saved in the h5 files,
    which may be normalised.
    """

    def __init__(self, folder_path="/eos/user/m/mamozzan/step2point/martina_test/metadata_p22_th45-135_ph79-109_en5-130/"):
        """
        Object that contains metadata for a dataset.
        Automatically constructed from what is found in the metadata folder.
        """
        self.metadata_folder = folder_path
        self._load_top_level()

    def _load_top_level(self):
        """
        Load data saved in the top level of the metadata folder
        in numpy files.

        Simple numpy arrays are loaded as attributes of this object
        of the same name as the file, with the .npy extension removed.

        Numpy files with pickled dictionaries are loaded with
        each key in the dictionary as an attribute of this object.
        """
        self.found_attrs = []
        for file in os.listdir(self.metadata_folder):
            if not file.endswith(".npy"):
                continue
            content = np.load(os.path.join(self.metadata_folder, file), allow_pickle=True)
            if content.dtype == "O":
                for key, value in content.item().items():
                    assert not hasattr(self, key)
                    self.found_attrs.append(key)
                    setattr(self, key, value)
            else:
                basename = os.path.basename(file)[: -len(".npy")]
                assert not hasattr(self, basename)
                self.found_attrs.append(basename)
                if str(content.dtype).startswith("<U"):
                    content = content.item()
                setattr(self, basename, content)

    def __repr__(self):
        text = str(self) + "\n"
        for attr in self.found_attrs:
            text += f"{attr}: {getattr(self, attr)}\n"
        return text

    def load_muon_map(self):
        """
        Load the muon map data from the muon_map subfolder of the metadata folder.
        Only done on request, as it's slightly larger.

        Creates the attributes muon_map_X, muon_map_Y, muon_map_Z, muon_map_E
        and also returns them.
        """
        data_dir = os.path.join(self.metadata_folder, "muon_map")

        self.muon_map_X = np.load(data_dir + "/X.npy")
        self.muon_map_Y = np.load(data_dir + "/Y.npy")
        self.muon_map_Z = np.load(data_dir + "/Z.npy")
        self.muon_map_E = np.load(data_dir + "/E.npy")

        return self.muon_map_X, self.muon_map_Y, self.muon_map_Z, self.muon_map_E

    @property
    def global_shower_axis_char(self):
        """
        The character representing the orientation of the shower axis,
        in the detector coordinate system.
        """
        expected_start = "hdf5:xyz==global:"
        if not self.orientation_global[: len(expected_start)] == expected_start:
            raise NotImplementedError(f"Don't know how to interpret global orientation {self.orientation_global}")
        global_order = self.orientation_global[len(expected_start) :]
        expected_start = "hdf5:xyz==local:"
        if not self.orientation[: len(expected_start)] == expected_start:
            raise NotImplementedError(f"Don't know how to interpret local orientation {self.orientation}")
        local_order = self.orientation[len(expected_start) :]
        char = global_order[local_order.index("z")]
        return char

    @property
    def global_shower_axis(self):
        """
        Orientation of the shower axis, in the detector coordinate system.
        """
        char = self.global_shower_axis_char
        vector = np.zeros(3)
        vector["xyz".index(char)] = 1
        return vector
