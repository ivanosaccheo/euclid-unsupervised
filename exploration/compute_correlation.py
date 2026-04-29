import polars as pl
import os
import numpy as np
from utility import load_utility as lu


savepath = os.path.expanduser("~/WORK/unsupervised/output")
os.makedirs(savepath, exist_ok=True)


filter_expression = ((pl.col("spurious_flag_euclid") <= 0.5) & 
                     (pl.col("det_quality_flag_euclid") == 0))
morph = ["mumax_minus_mag_euclid"]
extra =["object_id_euclid"]
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



df = lf.filter(filter_expression).select(photo + extra + morph).collect().fill_nan(None)
print(f"Keeping {df.height} sources", flush = True)
null_counts = df.null_count()
df = df.select((c for c in df.columns if null_counts[c][0]<df.height))


df_summary = (pl.concat([df.select(pl.all().mean().cast(pl.Float64)),
                         df.select(pl.all().std().cast(pl.Float64)),
                         df.select(pl.all().quantile(0.05).cast(pl.Float64)),
                         df.select(pl.all().quantile(0.16).cast(pl.Float64)),
                         df.select(pl.all().quantile(0.5).cast(pl.Float64)),
                         df.select(pl.all().quantile(0.84).cast(pl.Float64)),
                         df.select(pl.all().quantile(0.95).cast(pl.Float64)),

                         df.select(pl.all().is_not_null().mean().cast(pl.Float64)),
                         df.select(pl.all().is_not_null().sum().cast(pl.Float64))])
                        ).transpose(include_header = True, header_name = "feature",
                                    column_names=["mean", "std", 
                                                   "quant_05", "quant_16", "quant_50", "quant_84", "quant_95",
                                                   "frac_not_nan", "N_not_nan"])


correlation = df.drop_nulls()
print(f"{correlation.height} sources have all features", flush=True)
correlation = correlation.corr()


df_summary.write_csv(os.path.join(savepath, "summary_stats.csv"))
correlation.write_csv(os.path.join(savepath, "correlation.csv"))