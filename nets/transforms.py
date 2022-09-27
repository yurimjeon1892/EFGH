import sys
import os.path as osp
import numpy as np

import numba
from numba import njit, cffi_support

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'lib'))
import _khash_ffi

cffi_support.register_module(_khash_ffi)
khash_init = _khash_ffi.lib.khash_int2int_init
khash_get = _khash_ffi.lib.khash_int2int_get
khash_set = _khash_ffi.lib.khash_int2int_set
khash_destroy = _khash_ffi.lib.khash_int2int_destroy

np.set_printoptions(threshold=np.inf)

# ---------- BASIC operations ----------
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            return pic
        else:
            return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ---------- Build permutalhedral lattice ----------
@njit(numba.int64(numba.int64[:], numba.int64, numba.int64[:], numba.int64[:], ))
def key2int(key, dim, key_maxs, key_mins):
    """
    :param key: np.array
    :param dim: int
    :param key_maxs: np.array
    :param key_mins: np.array
    :return:
    """
    tmp_key = key - key_mins
    scales = key_maxs - key_mins + 1
    res = 0
    for idx in range(dim):
        res += tmp_key[idx]
        res *= scales[idx + 1]
    res += tmp_key[dim]
    return res


@njit(numba.int64[:](numba.int64, numba.int64, numba.int64[:], numba.int64[:], ))
def int2key(int_key, dim, key_maxs, key_mins):
    key = np.empty((dim + 1,), dtype=np.int64)
    scales = key_maxs - key_mins + 1
    for idx in range(dim, 0, -1):
        key[idx] = int_key % scales[idx]
        int_key -= key[idx]
        int_key //= scales[idx]
    key[0] = int_key

    key += key_mins
    return key


@njit
def advance_in_dimension(d1, increment, adv_dim, key):
    key_cp = key.copy()

    key_cp -= increment
    key_cp[adv_dim] += increment * d1
    return key_cp


class Traverse:
    def __init__(self, neighborhood_size, d):
        self.neighborhood_size = neighborhood_size
        self.d = d

    def go(self, start_key, hash_table_list):
        walking_keys = np.empty((self.d + 1, self.d + 1), dtype=np.long)
        self.walk_cuboid(start_key, 0, False, walking_keys, hash_table_list)

    def walk_cuboid(self, start_key, d, has_zero, walking_keys, hash_table_list):
        if d <= self.d:
            walking_keys[d] = start_key.copy()
            range_end = self.neighborhood_size + 1 if (has_zero or (d < self.d)) else 1

            for i in range(range_end):
                self.walk_cuboid(walking_keys[d], d + 1, has_zero or (i == 0), walking_keys, hash_table_list)
                walking_keys[d] = advance_in_dimension(self.d + 1, 1, d, walking_keys[d])
        else:
            hash_table_list.append(start_key.copy())


@njit
def build_it(pc1_num_points,
             d1, bcn_filter_size,
             pc1_keys_np,
             key_maxs, key_mins,
             pc1_lattice_offset,
             bcn_filter_offsets,
             pc1_blur_neighbors,
             last_pc1,
             assign_last):
    """
    :param pc1_num_points: int. Given
    :param d1: int. Given
    :param bcn_filter_size: int. Given. -1 indicates "do not filter"
    :param pc1_keys_np: (d1, N, d1) long. Given. lattice points coordinates
    :param key_maxs: (d1,) long. Given
    :param key_mins:
    :param pc1_lattice_offset: (d1, N) long. hash indices for pc1_keys_np
    :param bcn_filter_offsets: (bcn_filter_size, d1) long. Given.
    :param pc1_blur_neighbors: (bcn_filter_size, pc1_hash_cnt) long. hash indices. -1 means not in the hash table
    :param last_pc1: (d1, pc1_hash_cnt). permutohedral coordiantes for the next scale.
    :return:
    """
    # build hash table
    hash_table1 = khash_init()  # key to hash index
    key_hash_table1 = khash_init()  # hash index to key

    hash_cnt1 = 0
    for point_idx in range(pc1_num_points):
        for remainder in range(d1):
            key_int1 = key2int(pc1_keys_np[:, point_idx, remainder], d1 - 1, key_maxs, key_mins)
            hash_idx1 = khash_get(hash_table1, key_int1, -1)
            if hash_idx1 == -1:
                # insert lattice into hash table
                khash_set(hash_table1, key_int1, hash_cnt1)
                khash_set(key_hash_table1, hash_cnt1, key_int1)
                hash_idx1 = hash_cnt1
                if assign_last:
                    last_pc1[:, hash_idx1] = pc1_keys_np[:, point_idx, remainder]

                hash_cnt1 += 1
            pc1_lattice_offset[remainder, point_idx] = hash_idx1

    for hash_idx in range(hash_cnt1):
        pc1_int_key = khash_get(key_hash_table1, hash_idx, -1)
        pc1_key = int2key(pc1_int_key, d1 - 1, key_maxs, key_mins)

        if bcn_filter_size != -1:
            neighbor_keys = pc1_key + bcn_filter_offsets  # (#pts in the filter, d)
            for bcn_filter_index in range(bcn_filter_size):
                pc1_blur_neighbors[bcn_filter_index, hash_idx] = khash_get(hash_table1,
                                                                           key2int(neighbor_keys[bcn_filter_index, :],
                                                                                   d1 - 1,
                                                                                   key_maxs,
                                                                                   key_mins),
                                                                           -1)
    # destroy hash table
    khash_destroy(hash_table1)
    khash_destroy(key_hash_table1)
    return