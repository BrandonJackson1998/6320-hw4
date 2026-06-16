# Week 4 NYC TLC Trip-Duration Dataset Audit

## Source And Intended Use

The source data comes from the NYC Taxi and Limousine Commission Yellow Taxi trip record data.

- Dataset page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Default raw files:
  - https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
  - https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet
  - https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet
- Official Yellow Taxi data dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

NYC TLC states that trip records are collected from technology providers and include pickup/dropoff dates and times, location IDs, distances, fares, payment types, and passenger counts. TLC also warns that these data were not created by TLC and that TLC makes no representations about their accuracy.

For this course, the prepared dataset is used as a classroom training-failure and debugging exercise. It should not be treated as a production dispatch, pricing, staffing, routing, or transportation-policy model.

## Prediction Task

The prepared task is regression: predict `target_duration_seconds`, the duration of a yellow taxi trip, from pre-trip or early-trip fields.

The recommended training target is `target_log_duration`, which is `log1p(target_duration_seconds)`. Reporting can convert predictions back to seconds for MAE or other interpretable metrics.

## Preparation Summary

The preparation script reads only these raw columns:

- `tpep_pickup_datetime`
- `tpep_dropoff_datetime`
- `PULocationID`
- `DOLocationID`
- `trip_distance`
- `passenger_count`

It computes duration from pickup/dropoff timestamps, derives pickup time fields, renames location IDs, filters implausible rows, assigns a temporal split, and writes a small smoke-test file plus an optional full prepared file for CHPC.

The raw file is not committed to the repository. The full prepared file should be stored once on CHPC shared storage. The repository includes only the small smoke-test file and the transparent preparation/audit package.

## Filtering Rules

Rows are kept only when:

- pickup timestamp is inside the month named by the raw TLC file
- dropoff timestamp is after pickup timestamp
- `target_duration_seconds >= 60`
- `target_duration_seconds <= 14400`
- `trip_distance > 0`
- `trip_distance <= 100`
- pickup and dropoff location IDs are present
- `passenger_count` is present and between 1 and 6

These filters remove impossible or extreme records that would distract from the training-debugging focus of Week 4. They do not make the data perfect.

## Split And Leakage Guidance

The prepared file includes a `split` column assigned after sorting trips by pickup time:

- first 70 percent: `train`
- next 15 percent: `validation`
- final 15 percent: `test`

This time-based split is more responsible than a random row split for a deployment-like trip-duration task because validation and test rows occur later than training rows. Students should use the provided split column rather than creating a new random split.

Known leakage risks and mitigations:

- `tpep_dropoff_datetime` is used only to create the target and is not included as a model input.
- Fare, tip, toll, payment, and total-amount fields are excluded because they are post-trip or outcome-adjacent values.
- The `split` column is for data partitioning only and must not be used as a model input.
- Random splitting can overstate performance by mixing near-identical time/location patterns across train and validation; the provided temporal split avoids that specific issue.

## Responsible-Use Limitations

The data is operational and vendor-reported. It may contain inaccurate timestamps, distances, passenger counts, or location IDs. It reflects yellow taxi trips in New York City during one specific month and may not generalize to other months, policy periods, weather conditions, disruptions, transportation modes, or cities.

Location IDs can encode structural geography and service patterns. A model may perform differently across neighborhoods, times of day, or trip types. Students should avoid claims that a classroom model is fair, reliable, or operationally useful without additional evaluation.

## Feature Engineering Guidance

The prepared file includes low-level time fields such as `pickup_hour` and `pickup_day_of_week` but intentionally does not include hand-engineered fields such as `is_weekend` or `rush_hour_flag`. This keeps the default task focused on whether the model can learn useful time/location representations. Adding a simple engineered feature can be a valid debugging correction if the student explains the hypothesis and compares before/after behavior.
