# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "shortest.cpp":
    vector[vector[vector[int]]] extCall(vector[vector[int]] map, vector[pair[int,int]] starts)

def pyCallCpp(map, starts):
    return extCall(map=map, starts=starts)
