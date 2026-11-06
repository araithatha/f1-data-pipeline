# Data quality report

Built 2026-11-06T01:08:45Z

| Check | Result | Detail |
|---|---|---|
| dim_race primary key unique | PASS | 0 duplicate rows on ['race_id'] |
| dim_race keys not null | PASS | 0 rows with null keys |
| dim_driver primary key unique | PASS | 0 duplicate rows on ['driver_id'] |
| dim_driver keys not null | PASS | 0 rows with null keys |
| dim_constructor primary key unique | PASS | 0 duplicate rows on ['constructor_id'] |
| dim_constructor keys not null | PASS | 0 rows with null keys |
| fact_result primary key unique | PASS | 0 duplicate rows on ['race_id', 'driver_id'] |
| fact_result keys not null | PASS | 0 rows with null keys |
| fact_qualifying primary key unique | PASS | 0 duplicate rows on ['race_id', 'driver_id'] |
| fact_qualifying keys not null | PASS | 0 rows with null keys |
| fact_lap primary key unique | PASS | 0 duplicate rows on ['race_id', 'driver_number', 'lap_number'] |
| fact_lap keys not null | PASS | 0 rows with null keys |
| fact_pit_stop primary key unique | PASS | 0 duplicate rows on ['race_id', 'driver_id', 'stop_number'] |
| fact_pit_stop keys not null | PASS | 0 rows with null keys |
| fact_practice primary key unique | PASS | 0 duplicate rows on ['race_id', 'driver_id', 'session_name'] |
| fact_practice keys not null | PASS | 0 rows with null keys |
| winner 2023_01 | PASS | expected max_verstappen, got max_verstappen |
| winner 2024_01 | PASS | expected max_verstappen, got max_verstappen |
| winner 2024_02 | PASS | expected max_verstappen, got max_verstappen |
| winner 2024_03 | PASS | expected sainz, got sainz |
| winner 2024_06 | PASS | expected norris, got norris |
| winner 2019_21 | PASS | expected hamilton, got hamilton |
| exactly one winner per race | PASS | 0 races off |
| every result has a race | PASS | 0 orphan race_ids |
| lap times positive | PASS | 0 non-positive lap times |
| laps mapped to a driver | PASS | 100.00% mapped |
| laps with tyre compound | PASS | 99.75% have a compound |
| share of clean laps | PASS | 85.5% flagged clean |
| lap numbers within laps completed | PASS | 0 laps beyond the driver's race distance |
| races with Open-Meteo weather | PASS | 100.0% covered |
| Open-Meteo vs track rain sensor agree (2023+) | PASS | 89.9% of 89 races agree |
| races with pit-stop data | PASS | 99.6% of races |
