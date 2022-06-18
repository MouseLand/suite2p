import os
import suite2p 
from conftest import initialize_ops #Guarantees that tests and this script use the same ops

test_data_dir = 'test_data'

def generate_1p1c_expected_data():
    return 
def generate_1p1c1500_expected_data():
    return 

def main():
    #Create test_data directory if necessary
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
        print('Created test directory at ' + os.path.abspath(test_data_dir))
    ops = initialize_ops()
    return 
if __name__ == '__main__':
    main()
