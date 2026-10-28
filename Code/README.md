# Scripts

All paths are resolved from `paths.py` (`PACKAGE_ROOT` from `__file__`,
not cwd). Analysis scripts read `data/` and write `results/` and
`validation/`. They do not overwrite input CSVs.
`data/modeling_station_with_date.csv` is shipped. Reconstruct optionally
regenerates that file only; it does not overwrite `modeling_station.csv`,
it does not delete files, and it is not a required first step.

From the repository root: `python code/run_analysis.py`.

| Script | What it does |
|---|---|
| `run_analysis.py` | One command. Rebuilds Table 1, unadjusted means, adjusted OLS, all six Table S1 rows, recovery/hourly %, Figure 1–4, S1–S3, GAMM n-check, downwind audit, event-date sensitivity, the independent event-date refit, covariate provenance, and `validation/REPRODUCTION_CHECK.txt`. |
| `make_figure_1_maps.py` | Four-panel Figure 1 from `data/map/` (fire points, monitors, Montana outline). |
| `make_figure_s1_rx_distance_wind.py` | Figure S1 RX descriptive distance × wind grid (≤50 / ≤100 / ≤200 / ≤300 km) from `data/rx_distance_wind.csv`. |
| `make_figure_s2_distance_cutoffs.py` | Figure S2 distance caps 50–300 km on the primary east-sector sample. |
| `make_figure_s3_bearing_screens.py` | Figure S3 bearing screens, nested on 4,186 / 1,547, using `data/bearing_panel.csv.gz`. |
| `fit_adjusted_models.py` | Primary clustered OLS only. |
| `fit_event_date.py` | Independent event-date refit from `data/modeling_station_with_date.csv`. Writes `results/Table_event_date_coefficients.csv` and `validation/event_date_independent_refit.txt`. Does not replace 4.55. |
| `report_event_date_covariate_provenance.py` | Standalone provenance report (no OLS). Writes `validation/event_date_covariate_provenance.txt`. |
| `fit_event_id_sensitivity.py` | Name-year 4.55 vs Stage B event-date path (4.45). Writes `results/event_vs_name_year_comparison.txt`. Presents both estimates; does not replace 4.55. |
| `reconstruct_fire_date_from_stage_b.py` | Optional regeneration of the shipped `modeling_station_with_date.csv` from Stage B. Does not overwrite `modeling_station.csv`. |
| `fit_type_by_log1p_area.py` | Table S1 interaction. |
| `fit_distance_functional_form.py` | Table S2 distance-form AIC. Precomputes log(1+distance); does not use `np.log1p` inside the formula. |
| `analysis.py` | Shared loaders and the complete-case recipe, including `fillna(0)` for downwind. `load_modeling()` is the name-year 4.55 join (ignores Fire_Date). `load_modeling_event_correct()` reads the Stage B event+monitor table. `load_modeling_unique_keys()` is a name+year+monitor collapse (~4.47) only. |
| `paths.py` | `data/`, `results/`, `validation/` locations. |

`gamm/check_gamm_n.py` confirms 379,433 / 761 / 21 without fitting `bam()`
(sample counts only; coefficients, fitted smooths, and AIC were not refit).
`gamm/step05_fit_gamm_main.R` is the M4 fit. It reads the gzipped hourly table.
