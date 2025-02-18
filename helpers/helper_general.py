from functools import wraps
from time import time

def timing(f):
    """
    Decorator to measure execution time, adapted from
    # https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    # https://codereview.stackexchange.com/questions/169870/decorator-to-measure-execution-time-of-a-function
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f.__name__, f"Elapsed time: {end - start:.2f} sec")
        return result

    return wrapper
