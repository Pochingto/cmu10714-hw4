import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        reverted_shift_x = -shift_x
        reverted_shift_y = -shift_y

        img = np.roll(img, shift=reverted_shift_x, axis=0)
        img = np.roll(img, shift=reverted_shift_y, axis=1)

        if reverted_shift_x >= 0:
            img[:reverted_shift_x, :, :] = 0.0
        else:
            img[reverted_shift_x:, :, :] = 0.0
            
        if reverted_shift_y >= 0:
            img[:, :reverted_shift_y, :] = 0.0
        else:
            img[:, reverted_shift_y:, :] = 0.0

        return img
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device = None
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.iteration = 0
        self.n_samples = 0

        if self.shuffle:

            idxs = np.arange(len(self.dataset))
            np.random.shuffle(idxs)
            self.ordering = np.array_split(idxs, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        idxs = self.ordering[self.iteration]
        self.iteration += 1
        self.iteration %= len(self.ordering)

        self.n_samples += self.batch_size
        if self.n_samples > len(self.dataset):
            raise StopIteration

        batch = []
        # print(idxs)
        for i in range(len(self.dataset[0])):
            # data = [self.dataset[idx][i] for idx in idxs]
            # print(len(data))
            # print(len(data[0]))
            # print(self.dataset[0][i].shape)
            # print(Tensor([self.dataset[idx][i] for idx in idxs]).shape)
            batch.append(Tensor([self.dataset[idx][i] for idx in idxs], device=self.device))

        # for b in batch:
        #     print(b.shape)

        return tuple(batch)
        ### END YOUR SOLUTION

def parse_int_from_32bit_hex(binary):
    integer = 0
    index = len(binary) - 1
    for i, b in enumerate(binary):
        integer += b * (256 ** (index - i))

    return integer

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms=transforms)
        import gzip
        with gzip.open(image_filename,'rb') as fin:        
            magic_number = parse_int_from_32bit_hex(fin.read(4))
            num_imgs = parse_int_from_32bit_hex(fin.read(4))
            shape1 = parse_int_from_32bit_hex(fin.read(4))
            shape2 = parse_int_from_32bit_hex(fin.read(4))

            image_bytes = fin.read()
            images = np.frombuffer(image_bytes, dtype=np.uint8)
            images = images.reshape((num_imgs, shape1, shape2, 1))
            images = images/ 255.0
            # images = images.astype(np.float64)
            images = images.astype(np.float32)

        with gzip.open(label_filename,'rb') as fin:        
            magic_number = parse_int_from_32bit_hex(fin.read(4))
            num_labels = parse_int_from_32bit_hex(fin.read(4))

            labels_bytes = fin.read()
            labels = np.frombuffer(labels_bytes, dtype=np.uint8)

        self.images = images
        self.labels = labels

        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.images[index]), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms=transforms)
        data_files = []
        if train:
            data_files = [os.path.join(base_folder, f"data_batch_{i}") for i in range(1, 6)]
        else:
            data_files = [os.path.join(base_folder, f"test_batch")]

        X = np.array([]).reshape(0, 3072)
        y = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                X = np.concatenate([X, data_dict[b'data'] / 255.0], axis=0)
                y += data_dict[b'labels']

        self.X = X.reshape((-1, 3, 32, 32))
        self.y = y

        print("X shape: ", self.X.shape)
        print("len y: ", len(self.y))
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image = self.X[index]
        label = self.y[index]
        if self.transforms:
            image = self.apply_transforms(image)

        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        uid = None
        if not word in self.word2idx:
            uid = len(self.idx2word)
            self.word2idx[word] = uid
            self.idx2word.append(word)
        else:
            uid = self.idx2word.index(word)
        return uid
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        eos_id = self.dictionary.add_word("<eos>")
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                words = line.split()
                for word in words:
                    ids.append(self.dictionary.add_word(word))
                    ids.append(eos_id)
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    num_batch = len(data) // batch_size
    return np.array(data[:num_batch * batch_size], dtype=dtype).reshape((num_batch, batch_size))
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    seq_len = batches.shape[0]
    X = None
    y = None
    if i + bptt + 1 < seq_len:
        X = batches[i : i + bptt, :]
        y = batches[i + 1 : i + bptt + 1, :]
    elif i + 1 < seq_len:
        X = batches[i : -1, :]
        y = batches[i + 1:, :]
    else:
        raise Exception("index out of range (get_batch).")
    return Tensor(X, device=device, dtype=dtype), Tensor(y.flatten(), device=device, dtype=dtype)
    ### END YOUR SOLUTION