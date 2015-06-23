for feature in df_train.columns[:-1]:
    featureprior = df_train.groupby([feature])['WnvPresent'].sum()/df_train.groupby([feature])['WnvPresent'].count()
    df_train = df_train.join(featureprior, on=feature,rsuffix='_conditional_'+feature)
    df_test = df_test.join(featureprior, on=feature,rsuffix = '_conditional_'+feature)
