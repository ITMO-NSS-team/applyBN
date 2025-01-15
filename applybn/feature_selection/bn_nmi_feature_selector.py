
from pyitlib import discrete_random_variable as drv
import pandas as pd


class NMIFeatureSelectior:
  def __init__(self, threshhold = 0.5, bin_count=100, verboose=False):
    self.threshhold = threshhold
    self.bin_count = bin_count
    self.verboose = verboose


  def _discreticise(self, x, y=None):
    '''
    Prep data to feed into pyitlib drv.entropy
    Calc happenings of each event in descreticised feature
    '''
    if y is None:
      discr = [0] * (max(x) - min(x) + 1)
      for e in x:
        discr[int(e)] += 1
      return discr

    discr = dict()
    for e1, e2 in zip(x, y):
      if (e1,e2) not in discr:
        discr[(e1,e2)] = 0
      discr[(e1,e2)] += 1
    lenx = len(self._discreticise(x)) + 1
    leny = len(self._discreticise(y)) + 1
    res = [0] * (lenx * leny)
    for k, v in discr.items():
      res[(k[0] * leny + k[1])] = v
    return res


  def _NMI(self, a, b):
    '''
    Calculate Normalized Mutual Information (NMI) for pair of features
    '''
    if (a.name, None) not in self.entropy_map:
      self.entropy_map[(a.name, None)] = float(drv.entropy(self._discreticise(a)))
    if (b.name, None) not in self.entropy_map:
      self.entropy_map[(b.name, None)] = float(drv.entropy(self._discreticise(b)))
    if (a.name, b.name) not in self.entropy_map:
      self.entropy_map[(a.name, b.name)] = float(drv.entropy_joint(self._discreticise(a,b)))

    Ha = self.entropy_map[(a.name, None)]
    Hb = self.entropy_map[(b.name, None)]
    Hab = self.entropy_map[(a.name, b.name)]
    return (Ha + Hb - Hab) / (2 * max(Ha, Hb))

  def _prep_dataset(self, df, target):
    '''
    Prepare the dataset, discretize the fields, remove rows
    Return a ready-to-use dataset
    '''
    clean_df = pd.DataFrame()
    for col in df:
      if df[col].dtype in ['int', 'int32','int64']:
        # integers can be discreticised this way, without binning
        df_map = { l:i for i,l in enumerate(set(df[col].to_list()))}
        clean_df[col] = df[col].map(df_map)
      elif df[col].dtype in ['float', 'float64', 'double']:
        # discretisize to bin_count bins
        clean_df[col] = pd.cut(df[col].to_list(), bins=self.bin_count, labels=[i for i in range(self.bin_count)])
        clean_df[col] = clean_df[col].cat.add_categories(self.bin_count + 1).fillna(self.bin_count + 1)
      elif df[col].dtype in ['string', 'object']:
        # pandas cut does not work with string features, this is done manually
        # it is assumed that the user has removed the ID-like features, otherwise there will be a memory error later
        df_map = { l:i for i,l in enumerate(set(df[col].to_list()))}
        clean_df[col] = df[col].map(df_map)
    return clean_df

  def fit(self, df, target):
    clean_df = self._prep_dataset(df, target)
    if self.verboose:
      print('Dataset is ready')
    nmi_map = dict()
    self.entropy_map = dict()
    pre_features = []
    for col in clean_df:
      if col == target:
        continue
      if (col, target) not in nmi_map.keys():
        nmi_map[(col, target)] = self._NMI(clean_df[col], clean_df[target])
      nmi = nmi_map[(col, target)]
      if nmi > self.threshhold:
        pre_features.append(col)

    if self.verboose:
      print(f'First selection done, {len(pre_features)} selected', pre_features)

    if len(pre_features) == 1:
      return pre_features

    features = []
    for idx1, col1 in enumerate(pre_features):
      for idx2 in range(idx1, len(pre_features)):
        col2 = pre_features[idx2]
        if col1 == col2 or col1 == target or col2 == target:
          continue

        if (col1, col2) not in nmi_map:
          nmi_map[(col1, col2)] = self._NMI(clean_df[col1], clean_df[col2])

        nmi_tg_1 = nmi_map[(col1, target)]
        nmi_tg_2 = nmi_map[(col2, target)]
        nmi_col_12 = nmi_map[(col1, col2)]

        if (nmi_tg_1 > nmi_tg_2) and (nmi_col_12 > nmi_tg_2):
          if col1 not in features:
            features.append(col1)
        
    if self.verboose:
      print(f'Second selection done, reduced: {len(pre_features)} -> {len(features)}')

    self.entropy_map.clear()
    self.features_selected = features
    return self


  def transform(self):
    return self.features_selected
  

  def fit_transform(self, df, target):
    self.fit(df, target)
    return self.transform()
  