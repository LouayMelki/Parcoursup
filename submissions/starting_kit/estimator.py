from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def get_estimator():

    cat_cols = [
        'school_status','category_school','path','year','department','region_name'
    ]
    drop_cols = [
        "super_path","sub_path"
    ]

    categorical_transformer = Pipeline(steps=[
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols),
            ('drop cols', 'drop', drop_cols),
        ], remainder='passthrough')


    regressor = RandomForestRegressor(n_estimators=20, max_depth=15, max_features = 30)
    
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', regressor)
    ])

    return pipeline