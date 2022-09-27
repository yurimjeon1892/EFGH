import math
import numpy as np
import torch

from .transforms import Traverse, build_it

class GenerateData(object):

    def __init__(self, dim, scales_filter_map, device):
        self.d0 = dim
        self.d1 = self.d0 + 1
        self.scales_filter_map = scales_filter_map
        self.device = device

        elevate_left = torch.ones((self.d1, self.d0), dtype=torch.float32).triu()
        elevate_left[1:, ] += torch.diag(torch.arange(-1, -self.d0 - 1, -1, dtype=torch.float32))
        elevate_right = torch.diag(1. / (torch.arange(1, self.d0 + 1, dtype=torch.float32) *
                                         torch.arange(2, self.d0 + 2, dtype=torch.float32)).sqrt())
        self.expected_std = (self.d0 + 1) * math.sqrt(2 / 3)
        self.elevate_mat = torch.mm(elevate_left, elevate_right)  # what is this??????

        # (d+1,d)
        del elevate_left, elevate_right

        # canonical
        canonical = torch.arange(self.d0 + 1, dtype=torch.long)[None, :].repeat(self.d0 + 1, 1)
        # (d+1, d+1)
        for i in range(1, self.d0 + 1):
            canonical[-i:, i] = i - self.d0 - 1
        self.canonical = canonical
        # Y: Canonical simplex
        # tensor([[ 0,  1,  2,  3],
        #         [ 0,  1,  2, -1],
        #         [ 0,  1, -2, -1],
        #         [ 0, -3, -2, -1]])

        self.dim_indices = torch.arange(self.d0 + 1, dtype=torch.long)[:, None]
        # Y: dim_indices
        # tensor([[ 0],
        #         [ 1],
        #         [ 2],
        #         [ 3]])

        self.radius2offset = {}
        radius_set = set([item for line in self.scales_filter_map for item in line[1:] if item != -1])

        for radius in radius_set:
            hash_table = []
            center = np.array([0] * self.d1, dtype=np.long)
            traversal = Traverse(radius, self.d0)
            traversal.go(center, hash_table)
            self.radius2offset[radius] = np.vstack(hash_table)

        return

    def get_keys_and_barycentric(self, pc):
        """
        :param pc: (self.d0, N -- undefined)
        :return:
        """
        pc = pc[:self.d0, :]
        num_points = pc.size(-1)
        point_indices = torch.arange(num_points, dtype=torch.long)[None, :]

        pc = pc.type(torch.FloatTensor)
        # Y: Generate position vector by matmul (but idk what is std?)
        elevated = torch.matmul(self.elevate_mat, pc) * self.expected_std  # (d+1, N)

        # find 0-remainder
        greedy = torch.round(elevated / self.d1) * self.d1  # (d+1, N)

        el_minus_gr = elevated - greedy
        rank = torch.sort(el_minus_gr, dim=0, descending=True)[1]  # permutation order

        # the following advanced indexing is different in PyTorch 0.4.0 and 1.0.0
        # rank[rank, point_indices] = self.dim_indices  # works in PyTorch 0.4.0 but fail in PyTorch 1.x
        index = rank.clone()
        rank[index, point_indices] = self.dim_indices  # works both in PyTorch 1.x(has tested in PyTorch 1.2) and PyTorch 0.4.0
        del index

        remainder_sum = greedy.sum(dim=0, keepdim=True) / self.d1

        rank_float = rank.type(torch.float32)
        cond_mask = ((rank_float >= self.d1 - remainder_sum) * (remainder_sum > 0) + \
                     (rank_float < -remainder_sum) * (remainder_sum < 0)) \
            .type(torch.float32)
        sum_gt_zero_mask = (remainder_sum > 0).type(torch.float32)
        sum_lt_zero_mask = (remainder_sum < 0).type(torch.float32)
        sign_mask = -1 * sum_gt_zero_mask + sum_lt_zero_mask

        greedy += self.d1 * sign_mask * cond_mask
        rank += (self.d1 * sign_mask * cond_mask).type_as(rank)
        rank += remainder_sum.type(torch.long)

        # barycentric
        el_minus_gr = elevated - greedy
        greedy = greedy.type(torch.long)

        barycentric = torch.zeros((self.d1 + 1, num_points), dtype=torch.float32)
        barycentric[self.d0 - rank, point_indices] += el_minus_gr
        barycentric[self.d1 - rank, point_indices] -= el_minus_gr
        barycentric /= self.d1
        barycentric[0, point_indices] += 1. + barycentric[self.d1, point_indices]
        barycentric = barycentric[:-1, :]

        keys = greedy[:, :, None] + self.canonical[rank, :]  # (d1, num_points, d1)
        # rank: rearrange the coordinates of the canonical

        keys_np = keys.numpy()
        del elevated, greedy, rank, remainder_sum, rank_float, \
            cond_mask, sum_gt_zero_mask, sum_lt_zero_mask, sign_mask
        return keys_np, barycentric, el_minus_gr

    def get_filter_size(self, radius):
        return (radius + 1) ** self.d1 - radius ** self.d1

    def __call__(self, pc1):

        with torch.no_grad():

            # pc1 = torch.from_numpy(pc1)
            pc1 = pc1.type(torch.FloatTensor)
            last_pc1 = pc1.clone()

            generated_data = []
            pc1_num_points = last_pc1.size(-1)

            for idx, (scale, bcn_filter_raidus) in enumerate(self.scales_filter_map):

                last_pc1[:3, :] *= scale

                pc1_keys_np, pc1_barycentric, pc1_el_minus_gr = self.get_keys_and_barycentric(last_pc1)
                # keys: (d1, N, d1) [[:, point_idx, remainder_idx]], barycentric: (d1, N), el_minus_gr: (d1, N)

                key_maxs = pc1_keys_np.max(-1).max(-1)
                key_mins = pc1_keys_np.min(-1).min(-1)

                pc1_keys_set = set(map(tuple, pc1_keys_np.reshape(self.d1, -1).T))
                pc1_hash_cnt = len(pc1_keys_set)

                pc1_lattice_offset = np.empty((self.d1, pc1_num_points), dtype=np.int64)

                if bcn_filter_raidus != -1:
                    bcn_filter_size = self.get_filter_size(bcn_filter_raidus)
                    pc1_blur_neighbors = np.empty((bcn_filter_size, pc1_hash_cnt), dtype=np.int64)
                    pc1_blur_neighbors.fill(-1)
                    bcn_filter_offsets = self.radius2offset[bcn_filter_raidus]
                else:
                    bcn_filter_size = -1
                    pc1_blur_neighbors = np.zeros((1, 1), dtype=np.int64)
                    bcn_filter_offsets = np.zeros((1, 1), dtype=np.int64)

                if idx != len(self.scales_filter_map) - 1:
                    last_pc1 = np.empty((self.d1, pc1_hash_cnt), dtype=np.float32)
                else:
                    last_pc1 = np.zeros((1, 1), dtype=np.float32)

                build_it(pc1_num_points,
                         self.d1, bcn_filter_size,
                         pc1_keys_np,
                         key_maxs, key_mins,
                         pc1_lattice_offset,
                         bcn_filter_offsets,
                         pc1_blur_neighbors,
                         last_pc1,
                         idx != len(self.scales_filter_map) - 1)

                pc1_lattice_offset = torch.from_numpy(pc1_lattice_offset)

                if bcn_filter_size != -1:
                    pc1_blur_neighbors = torch.from_numpy(pc1_blur_neighbors)
                else:
                    pc1_blur_neighbors = torch.zeros(1, dtype=torch.long)

                if idx != len(self.scales_filter_map) - 1:
                    last_pc1 = torch.from_numpy(last_pc1)
                    last_pc1 /= self.expected_std * scale
                    last_pc1 = torch.matmul(self.elevate_mat.t(), last_pc1)
                    pc1_num_points = pc1_hash_cnt
                
                pc1_barycentric = torch.unsqueeze(pc1_barycentric, 0).to(self.device)
                pc1_el_minus_gr = torch.unsqueeze(pc1_el_minus_gr, 0).to(self.device)
                pc1_lattice_offset = torch.unsqueeze(pc1_lattice_offset, 0).to(self.device)
                pc1_blur_neighbors = torch.unsqueeze(pc1_blur_neighbors, 0).to(self.device)

                generated_data.append({'pc1_barycentric': pc1_barycentric,
                                       'pc1_el_minus_gr': pc1_el_minus_gr,
                                       'pc1_lattice_offset': pc1_lattice_offset,
                                       'pc1_blur_neighbors': pc1_blur_neighbors,
                                       'pc1_hash_cnt': pc1_hash_cnt,
                                       })

            return pc1, generated_data

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(scales_filter_map: {}\n'.format(self.scales_filter_map)
        format_string += ')'
        return format_string