import polars as pl
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score
from catboost import CatBoostClassifier
import numpy as np
from utility import load_utility as lu





filter_expression = ((pl.col("TARGETID_desi")>0) & (pl.col("spurious_flag_euclid") <= 0.5) & 
                     (pl.col("det_quality_flag_euclid") == 0))
morph = ["mumax_minus_mag_euclid"]
extra =["object_id_euclid", "SPECTYPE_desi"]
Nsteps = 500
remove_missing = False

savepath = os.path.expanduser("~/WORK/unsupervised/output")
os.makedirs(savepath, exist_ok=True)



file_path = "/scratch/extra/ELSA/ivano.saccheo2/DR1/EDFN_teresa_parquet"

lf= pl.scan_parquet(os.path.join(file_path, "*.parquet"))
schema = lf.collect_schema()
flux_columns = lu.get_templatefit_flux_columns(schema)
lf, mag_names  = lu.get_magnitudes(lf, flux_columns)
color_names = []
for fwhm in [1,2,3,4]:
    lf, names  = lu.get_colors(lf, lu.get_fwhm_flux_columns(schema, fwhm_values=[fwhm]))
    color_names.extend(names)

photo = mag_names + color_names 


df = lf.filter(filter_expression).select(photo + extra + morph).collect()
print(f"Keeping {df.height} sources", flush = True)
null_counts = df.fill_nan(None).null_count()
df = df.select((c for c in df.columns if null_counts[c][0]<df.height))
features = photo+morph
features = [f for f in features if f in df.columns]

if remove_missing:
    df = df.drop_nans(subset = features)
    print(f"Keeping {df.height} sources without missing features", flush = True)

for classifier_type in ["QSO", "STAR", "GALAXY", "multilabel"]:

    df = df.with_columns(lu._get_label_column_expr(classifier_type=classifier_type))
    y = df["label"].to_numpy()
    x = df[features].to_numpy()
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    rows = []

    for step in range(Nsteps):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=step)
        clf = CatBoostClassifier()
        clf.fit(x_train, y_train, verbose=False)
        predictions = clf.predict(x_test).squeeze()
        recall = recall_score(y_test, predictions, average = None)
        precision = precision_score(y_test, predictions, average = None)

        row = dict(zip(features, clf.get_feature_importance()))   
        for i, (r, p) in enumerate(zip(recall, precision)):
            row[f"recall_class_{i}"] = r
            row[f"precision_class_{i}"] = p

        rows.append(row)
        if step % 100 == 0:
            print(f"Step {step} done", flush=True)

    result_df = pd.DataFrame(rows)

    savename = f"feature_importance_{classifier_type}"
    if remove_missing:
        savename = savename + "_no_NaN.csv"
    else:
        savename = savename + "_all_sources.csv"

    result_df.to_csv(os.path.join(savepath, savename), index = False)
    
    print(f"{classifier_type} done", flush=True)

