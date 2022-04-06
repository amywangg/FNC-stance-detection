import sys
import Preprocessing


# Main driver code
if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Path to folder with data, test, train, validation: ", sys.argv[1])
        path = sys.argv[1]
        command = sys.argv[2]
        if command == 'preprocess':
            Preprocessing.preprocess(path)
