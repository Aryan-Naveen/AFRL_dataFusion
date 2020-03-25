def toTuple(arr):
    try:
        return tuple(toTuple(i) for i in arr)
    except TypeError:
        return arr