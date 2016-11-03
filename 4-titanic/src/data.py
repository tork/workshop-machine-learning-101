import os
import csv
import numpy as np

def titanic(path='data/titanic/titanic3.csv', norm_stats={}, ohot_stats={}, shuffle=True):
    # resolve titanic file path
    path_base = os.path.dirname(os.path.realpath(__file__))
    raw = RawCsvDataset('{}/../../{}'.format(path_base, path), norm_stats=norm_stats, ohot_stats=ohot_stats)

    # shuffling the dataset is good practice, in case the rows are sorted
    if shuffle:
        raw.shuffle()

    # survived is our target variable (what we want to model).
    # it is already typed as a binary integer value, and may be used as is
    y = np.ma.array(raw.data.survived)[:,None]
    # in python, nan != nan. the following line masks nan values
    y.mask = y != y

    ##############
    ### TASK 2 ###
    ##############
    # we need to select and preprocess variables from the raw csv data.
    # (the output variable survived is already taken care of above.)
    # for instance, we can normalize the fare variable like this:
    # fare = raw.normalize('fare')
    #
    # some variables should be expanded into one-hot vectors:
    # sexes = raw.to_one_hot('sex')
    #
    # in order to understand what variables to use, and how, you should
    # look through the csv file and understand how the values are formatted.
    # there is no right or wrong, but preprocessing will impact the learning.
    #
    # the variables are named as follows:
    # "pclass","survived","name","sex","age","sibsp","parch",
    # "ticket","fare", "cabin","embarked","boat","body","home.dest"
    # dataset description:
    # http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3info.txt
    #
    # once variables have been selected and processed, concatenate them
    # into a variable called x
    # x = np.ma.concatenate([fare, sexes], axis=1)

    fare = raw.normalize('fare')
    sexes = raw.to_one_hot('sex')
    # more variables?

    # remember to keep this concatenation up to date
    x = np.ma.concatenate([fare, sexes], axis=1)

    x.mask = y.mask = x.mask | y.mask
    return Dataset(x, y, norm_stats=raw.norm_stats, ohot_stats=raw.ohot_stats)


class Dataset(object):
    def __init__(self, x, y, norm_stats={}, ohot_stats={}):
        super(Dataset, self).__init__()
        assert x.shape[0] == y.shape[0]
        self.nvars = x.shape[1]
        self.nclasses = y.max()
        self.x = x
        self.y = y
        self.norm_stats = norm_stats
        self.ohot_stats = ohot_stats

    def split(self, ratio):
        len0 = int(len(self.x) * ratio)
        d0 = Dataset(self.x[0:len0], self.y[0:len0], self.norm_stats, self.ohot_stats)
        d1 = Dataset(self.x[len0:], self.y[len0:], self.norm_stats, self.ohot_stats)
        return d0, d1

# pandas is de facto for loading csv data.
# don't use numpy directly like this unless you have to.
#
# both numpy and pandas may decide to copy entire arrays during certain
# operations. for a tiny dataset like this however, it doesn't matter.
class RawCsvDataset(object):
    def __init__(self, path, delimiter=',', norm_stats={}, ohot_stats={}):
        super(RawCsvDataset, self).__init__()
        self.norm_stats = norm_stats
        self.ohot_stats = ohot_stats
        self.load_raw(path, delimiter=delimiter)

    def load_raw(self, path, delimiter=','):
        real_delim = chr(31)
        self.data = np.recfromcsv(
            (real_delim.join(i) for i in csv.reader(open(path))),
            delimiter=real_delim)
        if not self.data.shape:
            self.data = self.data[None]

    def shuffle(self):
        if self.data.shape[0] > 1:
            np.random.shuffle(self.data)

    # normalizes the variable name_var by ensuring
    # zero mean and unit variance (aka. std of 1)
    def normalize(self, name_var, dtype=np.float32):
        # mask nan values
        data = self.data[name_var].astype(dtype)
        mask = data != data
        data = np.ma.array(data, mask=mask)
        data[mask] = 0

        if name_var in self.norm_stats:
            stats = self.norm_stats[name_var]
        else:
            stats = self.norm_stats[name_var] = {
                'mean': data.mean(),
                'std': data.std() # (std := standard deviation)
            }
        data -= stats['mean']
        data /= stats['std']
        return data[:,None]

    # this method takes a discrete variable name_var.
    # first, each discrete value is mapped to an integer value.
    # next, the one-hot vector is created with the same number of rows
    # as the dataset, and as many columns as there are unique values.
    #
    # let's illustrate with the gender variable:
    # sex = { male, female }
    # the number of unique values is two: male and female.
    # the one-hot vector will therefore have two columns.
    # every value will be 0, except the row corresponding to each value.
    # for example, values [[male], [female], [female], [male]] would be transformed
    # to one-hot vector: [[1, 0], [0, 1], [0, 1], [1, 0]],
    # given that male maps to 0 and female maps to 1.
    def to_one_hot(self, name_var, dtype=np.float32):
        src = self.data[name_var]
        nsamples = src.shape[0]

        if name_var in self.ohot_stats:
            stats = self.ohot_stats[name_var]
        else:
            raw_classes = np.unique(src)
            nclasses = len(raw_classes)
            stats = self.ohot_stats[name_var] = {
                'nclasses': nclasses,
                'mapper': {raw_classes[i]: i for i in xrange(0, nclasses)}
            }

        nclasses = stats['nclasses']
        mapper = stats['mapper']

        dst = np.zeros([nsamples, nclasses], dtype=dtype)

        for i in xrange(0, nsamples):
            raw = src[i]
            if raw != None and raw == raw and raw != '':

                c = mapper[raw]
                dst[i,c] = 1

        return dst
