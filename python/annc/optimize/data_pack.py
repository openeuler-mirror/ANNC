from enum import Enum
from typing import List
import numpy as np


class DataFormat(Enum):
    mk = '(m, k)'
    kn = '(k, n)'
    k4m4 = '(m/4, k/4, m4, k4)'
    kn4 = '(n/4, k, n4)'


class MatrixTiling:
    axis: int = None  # 0, 1
    tile_size: int = None
    reverse: bool = None

    def __init__(self, axis: int, tile_size: int, reverse: bool = False):
        self.axis = axis
        self.tile_size = tile_size
        self.reverse = reverse


class FormageChange:
    # do not support data completion
    __align_completion__ = False

    def __init__(self, vector_len: int = 4):
        self.vector_len = vector_len

        self.call_packing_by_format_ = {
            (DataFormat.mk, DataFormat.k4m4): self.mk_to_k4m4,
            (DataFormat.kn, DataFormat.kn4): self.nk_to_kn4,
        }

    def run(self, data: np.ndarray, src_format: DataFormat,
            dst_format: DataFormat,
            tiling_info: List[MatrixTiling]) -> np.ndarray:
        blocks = self.do_tiling(data, tiling_info)
        result = self.do_packing(blocks, src_format, dst_format)
        return result.reshape(data.shape)

    def do_tiling(self, data: np.ndarray,
                  tiling_info: List[MatrixTiling]) -> List[np.ndarray]:
        blocks: List[np.ndarray] = [data]

        def block_tiling(block_: np.ndarray, axis_: int, tiling_num_: int,
                         reverse_: bool, new_blocks_: List[np.ndarray]):
            for i in range(tiling_num_):
                start = i * tiling.tile_size
                end = (i + 1) * tiling.tile_size
                dim_ = block_.shape[axis_]
                if dim_ < tiling.tile_size:
                    end = dim_
                elif dim_ % tiling.tile_size:
                    if i == tiling_num_ - 2:
                        res = dim_ - start
                        half = res // 2
                        docker_len = half // self.vector_len * self.vector_len
                        if res - docker_len > docker_len + 1:
                            docker_len += self.vector_len
                        end = start + docker_len
                    elif i == tiling_num_ - 1:
                        res = dim_ - start + tiling.tile_size
                        half = res // 2
                        docker_len = half // self.vector_len * self.vector_len
                        if res - docker_len > docker_len + 1:
                            docker_len += self.vector_len
                        start = dim_ - res + docker_len
                        end = dim_

                if reverse_:
                    start = dim_ - (i + 1) * tiling.tile_size
                    end = dim_ - i * tiling.tile_size

                if axis_ == 0:
                    new_blocks_.append(block_[start:end])
                elif axis_ == 1:
                    new_blocks_.append(block_[:, start:end])
                else:
                    raise TypeError(f'Unsupport tiling axis[{tiling.axis}]')

        for tiling in tiling_info:
            new_blocks = []
            for block in blocks:
                tiling_num = int(block.shape[tiling.axis] / tiling.tile_size)
                if block.shape[tiling.axis] % tiling.tile_size:
                    tiling_num += 1
                block_tiling(block, tiling.axis, tiling_num, tiling.reverse,
                             new_blocks)
            blocks = new_blocks
        return blocks

    def do_packing(self, blocks: List[np.ndarray], src_format: DataFormat,
                   dst_format: DataFormat) -> np.ndarray:
        pack_func = self.call_packing_by_format_[(src_format, dst_format)]
        result = np.array([], dtype=blocks[0].dtype)
        for block in blocks:
            result = np.concatenate([result, pack_func(block)], 0)
        return result

    def mk_to_k4m4(self, block: np.ndarray) -> np.ndarray:
        # convert lhs matrix[m,k] to [m/4,k/4,m4,k4]
        m_slices = self.generate_slices(block.shape[0], self.vector_len)
        k_slices = self.generate_slices(block.shape[1], self.vector_len)
        result = np.array([], dtype=block.dtype)
        m_loc = 0
        for m_slice in m_slices:
            k_loc = 0
            for k_slice in k_slices:
                slice_data = block[m_loc:m_loc + m_slice,
                                   k_loc:k_loc + k_slice]
                result = np.concatenate([result, slice_data.flatten()])
                k_loc += k_slice
            m_loc += m_slice
        return result

    def nk_to_kn4(self, block: np.ndarray) -> np.ndarray:
        # convert rhs matrix[k,n] to [n/4,k,n4]
        n_slices = self.generate_slices(block.shape[1], self.vector_len)
        result = np.array([], dtype=block.dtype)
        n_loc = 0
        for n_slice in n_slices:
            slice_data = block[:, n_loc:n_loc + n_slice]
            result = np.concatenate([result, slice_data.flatten()])
            n_loc += n_slice
        return result

    def generate_slices(self, dim: int, vec_len: int):
        slices = [vec_len] * int(dim / vec_len)
        boundary = dim % self.vector_len
        if boundary == 0:
            return slices

        if vec_len == 12:
            small_vec_len = 4
            while small_vec_len:
                while boundary >= small_vec_len:
                    slices.append(small_vec_len)
                    boundary -= small_vec_len                
                small_vec_len = int(small_vec_len / 2)
        else:
            small_vec_len = int(vec_len / 2)
            while small_vec_len:
                if boundary >= small_vec_len:
                    slices.append(small_vec_len)
                    boundary -= small_vec_len
                small_vec_len = int(small_vec_len / 2)
        return slices
