
import os
import torch
import hashlib
import tarfile
import requests
from tqdm import tqdm
import glob 
import cv2


class LFWDataset(torch.utils.data.Dataset):
    _DATA = (
        # images
        ("http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz", None),
        # segmentation masks as ppm
        ("https://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz",
         "3e7e26e801c3081d651c8c2ef3c45cfc"),
    )


    def __init__(self, base_folder, transforms, download=True, split_name: str = 'train'):
        super().__init__()
        self.base_folder = base_folder

        
        if download:
            self.download_resources(base_folder)

        self._dataset=LFWDataset._load_dataset(base_folder)
        self._transforms=transforms
        
        self.X = None
        self.Y = None
        #raise NotImplementedError("Not implemented yet")

    @staticmethod
    def get_image_and_groundtruth(path):
        samples_path=path + '\\lfw_funneled'
        segmentations_path=path + '\\parts_lfw_funneled_gt_images'

        images_paths=LFWDataset._get_image_paths(samples_path)

        groundtruth_pair_list=[]
        for image_path in images_paths:
            segmentation_path=LFWDataset._get_path_segmentation(image_path)
            if os.path.exists(segmentation_path):
                groundtruth_pair_list.append((image_path,segmentation_path))

        return groundtruth_pair_list

    @staticmethod
    def _get_path_segmentation(img_path):
        segmentation_name=img_path.split('\\')[8].split('.')[0]
        segmentation_path='D:\\UBB\\CVDL\\Project\\dataset\\lfw_dataset\\parts_lfw_funneled_gt_images\\' + segmentation_name + '.ppm'
        return segmentation_path


    @staticmethod
    def _get_image_paths(dir):
        paths=[]
        #search recursevly for all jpg files in the dir
        for file_path in glob.glob(dir + '\**\*.jpg', recursive=True):
            paths.append(file_path)
        return paths

    @staticmethod
    def _get_inputs_and_segmentations(inputs_segmentation_paths):
        inputs=[]
        for input_path, segmentation_path in inputs_segmentation_paths:
            sample=cv2.imread(input_path)
            sample=cv2.cvtColor(sample,cv2.COLOR_BGR2RGB)

            segmentation=cv2.imread(segmentation_path)
            segmentation=cv2.cvtColor(segmentation,cv2.COLOR_BGR2RGB)

            inputs.append((sample,segmentation))
        return inputs




    @staticmethod
    def _load_dataset(path):
        img_and_truths_paths=LFWDataset.get_image_and_groundtruth(path)
        inputs_and_segmentations=LFWDataset._get_inputs_and_segmentations(img_and_truths_paths)
        return inputs_and_segmentations

    def __len__(self):
        return len(self._dataset) 
    

    def __getitem__(self, idx):
        
        # TODO your code here: return the idx^th sample in the dataset: image, segmentation mask
        # TODO your code here: if necessary apply the transforms

        item=self._dataset[idx]
        if self._transforms:
            item=(self._transforms(item[0]),self._transforms(item[1]))
        return item[0],item[1]

    def download_resources(self, base_folder):
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        self._download_and_extract_archive(url=LFWDataset._DATA[1][0], base_folder=base_folder,
                                           md5=LFWDataset._DATA[1][1])
        self._download_and_extract_archive(url=LFWDataset._DATA[0][0], base_folder=base_folder, md5=None)

    def _download_and_extract_archive(self, url, base_folder, md5) -> None:
        """
          Downloads an archive file from a given URL, saves it to the specified base folder,
          and then extracts its contents to the base folder.

          Args:
          - url (str): The URL from which the archive file needs to be downloaded.
          - base_folder (str): The path where the downloaded archive file will be saved and extracted.
          - md5 (str): The MD5 checksum of the expected archive file for validation.
          """
        base_folder = os.path.expanduser(base_folder)
        filename = os.path.basename(url)

        self._download_url(url, base_folder, md5)
        archive = os.path.join(base_folder, filename)
        print(f"Extracting {archive} to {base_folder}")
        self._extract_tar_archive(archive, base_folder, True)

    def _retreive(self, url, save_location, chunk_size: int = 1024 * 32) -> None:
        """
            Downloads a file from a given URL and saves it to the specified location.

            Args:
            - url (str): The URL from which the file needs to be downloaded.
            - save_location (str): The path where the downloaded file will be saved.
            - chunk_size (int, optional): The size of each chunk of data to be downloaded. Defaults to 32 KB.
            """
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(save_location, 'wb') as file, tqdm(
                    desc=os.path.basename(save_location),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    bar.update(len(data))

            print(f"Download successful. File saved to: {save_location}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def _download_url(self, url: str, base_folder: str, md5: str = None) -> None:
        """Downloads the file from the url to the specified folder

        Args:
            url (str): URL to download file from
            base_folder (str): Directory to place downloaded file in
            md5 (str, optional): MD5 checksum of the download. If None, do not check
        """
        base_folder = os.path.expanduser(base_folder)
        filename = os.path.basename(url)
        file_path = os.path.join(base_folder, filename)

        os.makedirs(base_folder, exist_ok=True)

        # check if the file already exists
        if self._check_file(file_path, md5):
            print(f"File {file_path} already exists. Using that version")
            return

        print(f"Downloading {url} to file_path")
        self._retreive(url, file_path)

        # check integrity of downloaded file
        if not self._check_file(file_path, md5):
            raise RuntimeError("File not found or corrupted.")

    def _extract_tar_archive(self, from_path: str, to_path: str = None, remove_finished: bool = False) -> str:
        """Extract a tar archive.

        Args:
            from_path (str): Path to the file to be extracted.
            to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
                used.
            remove_finished (bool): If True , remove the file after the extraction.
        Returns:
            (str): Path to the directory the file was extracted to.
        """
        if to_path is None:
            to_path = os.path.dirname(from_path)

        with tarfile.open(from_path, "r") as tar:
            tar.extractall(to_path)

        if remove_finished:
            os.remove(from_path)

        return to_path

    def _compute_md5(self, filepath: str, chunk_size: int = 1024 * 1024) -> str:
        with open(filepath, "rb") as f:
            md5 = hashlib.md5()
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

    def _check_file(self, filepath: str, md5: str) -> bool:
        if not os.path.isfile(filepath):
            return False
        if md5 is None:
            return True
        return self._compute_md5(filepath) == md5
