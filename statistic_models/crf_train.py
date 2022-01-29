#!/usr/bin/env python

from statistic_models.crf import LinearChainCRF

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("datafile", help="data file for training input", default="D:\data\data_crf\chunking_small\\small_train.data", required=False)
    # parser.add_argument("modelfile", help="the model file name. (output)", default="D:\\tmp\crf\\model.json", required=False)
    #
    # args = parser.parse_args()

    crf = LinearChainCRF()
    datafile = "D:\data\data_crf\chunking_small\\small_train.data"
    modelfile = "D:\\tmp\crf\\model.json"
    crf.train(datafile, modelfile)
