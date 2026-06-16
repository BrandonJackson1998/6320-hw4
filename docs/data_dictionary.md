# Week 4 NYC TLC Prepared Data Dictionary

Prepared file: `week04_tlc_trip_duration_smoke_10k.parquet` locally, and the matching full prepared Parquet file on CHPC.

## Columns

| Column | Type | Role | Description |
|---|---|---|---|
| `target_duration_seconds` | float | Target | Trip duration in seconds, computed as dropoff timestamp minus pickup timestamp. |
| `target_log_duration` | float | Training target | `log1p(target_duration_seconds)`, included because duration is right-skewed and raw-second regression can be unstable. |
| `pickup_location_id` | integer categorical | Input | NYC TLC pickup taxi zone ID from raw `PULocationID`. |
| `dropoff_location_id` | integer categorical | Input | NYC TLC dropoff taxi zone ID from raw `DOLocationID`. |
| `pickup_hour` | integer categorical | Input | Hour of day from pickup timestamp, 0-23. |
| `pickup_day_of_week` | integer categorical | Input | Day of week from pickup timestamp, Monday = 0 through Sunday = 6. |
| `pickup_day_of_month` | integer numeric or categorical | Input | Day of January from pickup timestamp, 1-31. |
| `trip_distance` | float | Input | Driver- or meter-reported trip distance from the raw trip record. |
| `passenger_count` | float | Input | Passenger count from the raw trip record after filtering to values 1-6. |
| `split` | categorical string | Split control | One of `train`, `validation`, or `test`, assigned by temporal order. Do not use as a model input. |

## Deliberately Excluded Raw Fields

The prepared dataset excludes `tpep_dropoff_datetime` as a model input because it defines the target. It also excludes fare, tip, toll, payment, and total-amount fields because they are post-trip or outcome-adjacent values that are not safe pre-trip prediction inputs.

## Feature Engineering Choice

The package provides low-level time fields such as `pickup_hour` and `pickup_day_of_week` rather than hand-engineered flags such as `is_weekend` or `rush_hour`. This lets a neural network with embeddings or categorical representations learn those patterns from the data. Students may add simple engineered indicators as an explicit debugging or correction experiment, but the default prepared file keeps these choices visible.
