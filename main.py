import suite2p
import numpy as np

def main():
    ops = np.load('ops.npy')
    ops = ops.item()
    suite2p.main(ops)

if __name__ == '__main__':
    main()
