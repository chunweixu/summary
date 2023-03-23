import time

def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Function {} took {} seconds to execute".format(func.__name__, end_time - start_time))
        return result
    return wrapper

def read_file(path):
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        return [_.strip() for _ in lines]