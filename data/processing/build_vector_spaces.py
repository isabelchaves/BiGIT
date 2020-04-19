import nmslib


def build_vector_space(data: dict, vector_space: str, vector_method: str):
    data_vs = nmslib.init(method=vector_method, space=vector_space)
    data_vs.addDataPointBatch(list(data.values()))
    data_vs.createIndex()

    return data_vs
