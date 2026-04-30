import os
import polars as pl 
from itertools import combinations
import numpy as np



BAND_ORDER = [
    "u", "g", "r", "i", "z",
    "VIS",
    "Y", "J", "H", "NIR",
    "W1", "W2", "W3", "W4" ]


def _get_magnitudes_expression(flux_columns, prefix = "mag"):
    magnitude_names = get_magnitudes_names(flux_columns, prefix = prefix)
    expression = [(-2.5 * pl.col(f).log10() + 23.9).alias(m) 
                  for (f,m) in zip(flux_columns, magnitude_names)]
    return expression, magnitude_names

def _get_errors_magnitudes_expression(flux_columns, error_columns, prefix = "magerr"):
    error_names = get_magnitudes_names(flux_columns, prefix = prefix)
    expression = [((2.5/np.log(10)) * pl.col(err) / pl.col(f)).alias(m) 
                  for (f, err, m) in zip(flux_columns, error_columns, error_names)]
    return expression, error_names

def _replace_flux_prefix(name, prefix):
    if not name.startswith("flux_"):
        raise ValueError(f"Unexpected column name: {name}")
    if prefix == "":
        return name[len("flux_"):]  # rimuove solo "flux_"
    return name.replace("flux_", f"{prefix}_", 1)

def _band_key(col, band_order=BAND_ORDER):
    for i, band in enumerate(band_order):
        if f'_{band.lower()}_' in col.lower():
            return i
    return len(band_order)

def _sort_flux_columns_by_wavelength(flux_columns, band_order=BAND_ORDER):
    
    return sorted(flux_columns, key=lambda col: _band_key(col, band_order))


def _get_label_column_expr(classifier_type):

    if classifier_type in ["GALAXY", "STAR", "QSO"]:
        expression = (pl.when(pl.col("SPECTYPE_desi") == classifier_type).then(1)
                            .otherwise(0).alias("label"))
    elif classifier_type == "multilabel":
        
        expression = (pl.when(pl.col("SPECTYPE_desi") == "STAR").then(0)
                        .when(pl.col("SPECTYPE_desi") == "GALAXY").then(1)
                        .when(pl.col("SPECTYPE_desi") == "QSO").then(2)
                        .otherwise(-1).alias("label"))
    else:
        raise ValueError("Wrong classifier specified")
    
    return expression


def get_color_pairs(flux_columns1, flux_columns2=None, band_order=BAND_ORDER):
    if flux_columns2 is None:
        flux_columns1 = _sort_flux_columns_by_wavelength(
            flux_columns1, band_order=band_order
        )
        return list(combinations(flux_columns1, r=2))

    return [
        tuple(sorted((f1, f2), key=lambda col: _band_key(col, band_order=band_order)))
        for f1, f2 in zip(flux_columns1, flux_columns2)
    ]

def get_magnitudes(df, flux_columns, 
                   error_columns = None, 
                   prefix = "mag", 
                   return_names = True):
    if isinstance(flux_columns, str):
        flux_columns =[flux_columns]
    expression, names  = _get_magnitudes_expression(flux_columns, prefix = prefix)
    if error_columns is not None:
        expression_err, names_err = _get_errors_magnitudes_expression(flux_columns, 
                                                                      error_columns, 
                                                                      prefix =prefix+"err")
        expression = expression + expression_err
        names = names + names_err
    if return_names:
        return df.with_columns(expression), names
    return df.with_columns(expression)

def get_colors(df, flux_columns1, flux_columns2 = None, return_names = True,
               prefix = "", 
               band_order = BAND_ORDER):
    
    pairs = get_color_pairs(flux_columns1,flux_columns2=flux_columns2,
                            band_order=band_order,)
    names = get_color_names_from_pairs(pairs, prefix=prefix)

    expressions = [(-2.5 * (pl.col(f1) / pl.col(f2)).log10()).alias(name)
            for (f1, f2), name in zip(pairs, names)]
    
    if return_names:
        return df.with_columns(expressions), names
    return df.with_columns(expressions)


def get_magnitudes_names(flux_columns, prefix="mag"):
    return [_replace_flux_prefix(col, prefix) for col in flux_columns]


def get_color_names_from_pairs(pairs, prefix=""):
    return [
        f"{_replace_flux_prefix(f1, prefix)}-{_replace_flux_prefix(f2, prefix)}"
        for f1, f2 in pairs
    ]


def get_float_columns(schema):
    return [name for name, dtype in schema.items() if dtype.is_float()]

def get_numeric_columns(schema):
    return [name for name, dtype in schema.items() if dtype.is_numeric()]

def get_error_columns(schema):
    return [name for name in schema.keys() if "err_" in name]


def get_templatefit_flux_columns(schema, ebv_corrected=True):
    cols = []

    for name in schema.names():
        if (
            "flux_" in name
            and "_euclid" in name
            and "_templfit" in name
            and "_total" in name
        ):
            if ebv_corrected and "_ebv_" in name:
                cols.append(name)
            elif not ebv_corrected and "_ebv_" not in name:
                cols.append(name)

    return cols


def get_fwhm_flux_columns(schema, fwhm_values=None):
    cols = []

    for name in schema.names():
        if (
            "flux_" in name
            and "_euclid" in name
            and "_total" not in name
        ):
            if fwhm_values is None:
                if "fwhm" in name:
                    cols.append(name)
            else:
                for f in fwhm_values:
                    if f"{f}fwhm" in name:
                        cols.append(name)
                        break
    return cols

def get_fluxerror_columns(schema, flux_columns):
    error_columns = [f.replace("flux_", "fluxerr_") for f in flux_columns]
    missing = [col for col in error_columns if col not in schema.names()]
    if missing:
        raise KeyError(f"Missing column in the schems: {missing}")
    return error_columns


####This is just a convenient function
def load_data(
    file_path = "/scratch/extra/ELSA/ivano.saccheo2/DR1/EDFN_teresa_parquet",
    *,
    fwhm_values = (1,2,3,4),
    add_magnitudes=True,
    add_colors=True,
    add_errors = True,
    filter_expressions = ((pl.col("spurious_flag_euclid") <= 0.5) & 
                         (pl.col("det_quality_flag_euclid") == 0)),
    extra_columns = ("object_id_euclid",),
    morphology_columns = ("mumax_minus_mag_euclid",),
    ):
                          
    lf = pl.scan_parquet(os.path.join(file_path, "*.parquet"))
    schema = lf.collect_schema()
    mag_names = []
    color_names = []
    if filter_expressions is not None:
        lf = lf.filter(filter_expressions)
    if add_magnitudes:
        flux_columns = get_templatefit_flux_columns(schema)
        error_columns = None
        if add_errors:
            error_columns = get_fluxerror_columns(schema, flux_columns)
        lf, mag_names = get_magnitudes(lf, flux_columns, error_columns=error_columns)
    if add_colors:
        for fwhm in fwhm_values:
            fwhm_flux_columns = get_fwhm_flux_columns(schema,fwhm_values=[fwhm])
            lf, names = get_colors(lf, fwhm_flux_columns)
            color_names.extend(names)
    morphology_columns = list(morphology_columns or [])
    extra_columns = list(extra_columns or [])

    feature_columns = mag_names + color_names + morphology_columns

    df = lf.select(extra_columns +  feature_columns).collect()
    null_counts = df.fill_nan(None).null_count()
    df = df.select((c for c in df.columns if null_counts[c][0]<df.height)) 
    feature_columns = [f for f in feature_columns if f in df.columns]

    return df, feature_columns