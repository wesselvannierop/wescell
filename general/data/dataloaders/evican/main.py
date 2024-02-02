from pathlib import Path

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Local imports
from general.data.dataloaders.generic import GenericDataset


class EvicanDataset(GenericDataset):
    def __init__(self, *args, **kwargs):
        """Use the GenericDataset class to load the images and targets from the EVICAN dataset."""
        super().__init__(*args, **kwargs)
        assert self.split != "test", "Test split is not implemented (requires pycocotools)"

    def setup(self):
        # Get the paths to the images and targets of EVICAN
        dataset_path = Path(self.root) / "Images" / f"EVICAN_{self.split}2019"
        self.files = self.get_cell_files(dataset_path)
        masks_path = Path(self.root) / "Masks" / f"EVICAN_{self.split}_masks" / "Cells"
        self.mask_files = self.get_cell_files(masks_path)
        self.assertions()
        return self

    @staticmethod
    def get_cell_files(path):
        files = [file for file in path.iterdir() if not "Background" in file.stem]
        return sorted(files)
