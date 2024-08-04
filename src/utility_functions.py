def preprocess_data(df, test=False):
    
    lithology_keys = {30000: 'Sandstone',
                 65030: 'Sandstone/Shale',
                 65000: 'Shale',
                 80000: 'Marl',
                 74000: 'Dolomite',
                 70000: 'Limestone',
                 70032: 'Chalk',
                 88000: 'Halite',
                 86000: 'Anhydrite',
                 99000: 'Tuff',
                 90000: 'Coal',
                 93000: 'Basement'}
    
    df_num = df.select_dtypes(include=[int, float])
    df_cat = df.select_dtypes(include=[object])
    
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].fillna('unkwn')
    
    if test:
        return df_num, df_cat
    else:
        target = df_num['FORCE_2020_LITHOFACIES_LITHOLOGY']
        df_num = df_num.drop(['FORCE_2020_LITHOFACIES_LITHOLOGY', 'FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)        
        
        return df_num, df_cat, target.map(lithology_keys)