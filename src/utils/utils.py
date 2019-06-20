import _pickle as cPickle
import argparse as arg
import os


def get_args():
    argparse = arg.ArgumentParser(description=__doc__)

    argparse.add_argument(
        '-c', '--config',
        metavar='c',
        help='Config file path')

    argparse.add_argument(
        '-i', '--img_path',
        metavar='i',
        help='image path')

    argparse.add_argument(
        '-t', '--test_path',
        metavar='t',
        help='test images path')
    args = argparse.parse_args()
    return args


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def print_predictions(paths, predictions):
    if paths.shape[0] != predictions.shape[0]:
        print("inputs and predictions shapes are not equal.")
        return
    print("Predictions:\n")
    for i in range(paths.shape[0]):
        print("   Image:%s|Prediction:%d\n" % (os.path.basename(paths[i]), predictions[i]))
