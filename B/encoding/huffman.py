# -*- coding: utf-8 -*-
from __future__ import print_function
from dahuffman import HuffmanCodec
from . import runlength as rle
import numpy as np
import collections
import time
import lz4framed
import torch
from ast import literal_eval as make_tuple


def encode(text):
    frequencies = collections.Counter(text)
    codec = HuffmanCodec.from_frequencies(frequencies)
    return codec.encode(text), frequencies

def decode(text, frequencies):
    codec = HuffmanCodec.from_frequencies(frequencies)
    return codec.decode(text)

def fullencode(text):
    return lz4framed.compress(text)

def fulldecode(text):
    return lz4framed.decompress(text)

def encode_with_indexs(ny):
    shape = ny.shape
    indexes = np.flatnonzero(ny).astype(np.uint32)
    #indexes = np.transpose(np.nonzero(ny)).astype(np.uint32)
    #print('indexes type: ', indexes.dtype)
    vals = ny.flatten()[indexes]
    #vals = ny[indexes]
    by = vals.tobytes()
    compressed = by#fullencode(by)
    by = indexes.tobytes()
    indexes_compressed = by#fullencode(by)
    seperator = b'-+-+'
    final_compressed = b''.join([compressed, seperator, indexes_compressed, seperator, str(shape)])
    return final_compressed 

def decode_with_indexs(text, gpu_mem=None):
    seperator = b'-+-+'
    items = text.split(seperator)
    vals = items[0] #fulldecode(items[0])
    vals = np.frombuffer(vals, dtype=np.float32)
    indexes = items[1] #fulldecode(items[1])
    indexes = np.frombuffer(indexes, dtype=np.int32)
    shape = make_tuple(items[2])
    size = 1
    #print('shape: ', shape)
    for s in list(shape):
        size *= s
    if gpu_mem is not None:
        gpu_indexes = torch.from_numpy(indexes).cuda().type(torch.cuda.LongTensor)
        gpu_vals = torch.from_numpy(vals).cuda()
        gpu_mem.zero_()
        gpu_mem[gpu_indexes] = gpu_vals
        return gpu_mem
    ny = np.zeros(size).astype(np.float32)
    ny[indexes] = vals
    return ny

def test():
    def _check(a, ra):
        ret = np.array_equal(a, ra)
        print('Equal: ', ret)
    shape = (1024, 1024)
    a = np.random.uniform(-1, 1, shape).astype(np.float32)
    s = 0.95
    zeros = np.random.binomial(1, 1-s, 1024*1024)
    #gpu_mem = torch.zeros(1024*1024, dtype=torch.float32).cuda()
    gpu_mem = None
    st = time.time()
    zeros = zeros.reshape(a.shape)
    a *= zeros
    #a = a.astype(np.int8)
    nnz = np.count_nonzero(a)
    print('nnz: ', nnz)
    bytes = a.tobytes()
    olen = len(bytes)
    print('Original len: ', len(bytes))
    #bytes = rle.encode(bytes)
    #print('RLE Compressed len: ', len(bytes))
    #bytes, frequencies = encode(bytes)
    #codec = HuffmanCodec.from_data(bytes)
    #bytes = codec.encode(bytes)
    #bytes = fullencode(bytes)
    bytes = encode_with_indexs(a)
    print('Compressed len: ', len(bytes))
    print('Compressed rate: ', olen*1./len(bytes))
    print('Compress time used: ', time.time()-st)
    #bytes = decode(bytes, frequencies)
    #rbytes = rle.decode(bytes)
    #rbytes = fulldecode(bytes)
    st = time.time()
    rbytes = decode_with_indexs(bytes, gpu_mem)
    print('decode: ', len(rbytes))
    if gpu_mem is not None:
        ra = rbytes.cpu().numpy()
        ra = ra.reshape(a.shape)
    else:
        ra = np.frombuffer(rbytes, dtype='float32').reshape(a.shape)
        t = torch.from_numpy(ra).cuda()
    print('decode time used: ', time.time()- st)
    _check(a, ra)


if __name__ == '__main__':
    test()
