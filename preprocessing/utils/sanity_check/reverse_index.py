import numpy as np

if __name__=='__main__':

    x = np.array([57, 3, 4, 1, 15, 14, 17, 21], dtype=np.int64)
    reverse_indices = np.argsort(np.argsort(x))
    x_sorted = np.sort(x)
    x_recovered = x_sorted[reverse_indices]

    print(f'original ={x}')
    print(f'sorted 1 ={x_sorted}')
    print(f'recovered={x_recovered}')
