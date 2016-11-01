import os
import csv
import numpy as np

def titanic(path='data/titanic/titanic3.csv', norm_stats={}, ohot_stats={}, shuffle=True):
    # column_index:variable_name
    # 0:pclass, 1:survived, 3:sex, 4:age, 5:sibsp, 6:parch, 8:fare, 9:cabin
    path_base = os.path.dirname(os.path.realpath(__file__))
    raw = RawCsvDataset('{}/../../{}'.format(path_base, path), norm_stats=norm_stats, ohot_stats=ohot_stats)
    if shuffle:
        raw.shuffle()

    # survived is our target variable (what we want to model).
    # it is already typed as a binary integer value, and may be used as is
    y = np.ma.array(raw.data.survived)
    y.mask = y != y

    # certain variables need normalization before use
    age = raw.normalize('age')
    sibsp = raw.normalize('sibsp')
    parch = raw.normalize('parch')
    fare = raw.normalize('fare')

    # discrete variables should often be expanded to a one-hot vector.
    # this helps the network discriminate input
    sex = raw.to_one_hot('sex')
    pclasses = raw.to_one_hot('pclass').transpose()
    # assuming there are two genders only, we could also use these values as is
    sexes = raw.to_one_hot('sex').transpose()

    # we have cabin numbers for each passenger, as strings.
    # perhaps we can use it somehow? eg. extract floor number or placement
    # cabin = ?

    x = np.ma.concatenate([[sibsp], [parch], [fare], pclasses, sexes]).transpose()
    # x = np.ma.concatenate([[age], [sibsp], [parch], [fare], pclasses, sexes]).transpose()
    x.mask = y.mask = x.mask | y.mask
    return Dataset(x, y[:,None], norm_stats=raw.norm_stats, ohot_stats=raw.ohot_stats)


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

    def normalize(self, name_var, dtype=np.float32):
        # mask nan values
        data = self.data[name_var].astype(dtype)
        mask = data != data
        data = np.ma.array(data, mask=mask)
        data[~mask] = 0
        data.mask = mask

        if name_var in self.norm_stats:
            stats = self.norm_stats[name_var]
        else:
            stats = self.norm_stats[name_var] = {
                'mean': data.mean(),
                'std': data.std()
            }
        data -= stats['mean']
        data /= stats['std']
        return data

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

        # the + 1 reserves a special class for missing values.
        # a more commonly used approach is to filter such values out
        dst = np.zeros([nsamples, nclasses + 1], dtype=dtype)
        nan = float('nan')

        for i in xrange(0, nsamples):
            raw = src[i]
            if raw != None and raw != nan and raw != '':
                c = mapper[raw]
            else:
                c = nclasses
            dst[i][c] = 1

        return dst
