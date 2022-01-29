#!/usr/bin/env python

from statistic_models.crf import LinearChainCRF

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("datafile", help="data file for testing input")
    # parser.add_argument("modelfile", help="the model file name.")
    #
    # args = parser.parse_args()

    crf = LinearChainCRF()
    datafile = "D:\data\data_crf\chunking_small\\small_test.data"
    modelfile = "D:\\tmp\crf\\model.json"
    crf.load(modelfile)
    crf.test(datafile)
