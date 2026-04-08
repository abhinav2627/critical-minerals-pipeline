-- stg_geochemistry.sql
-- Staging model for Project 1 geochemistry features
-- Standardises column names, casts types, filters invalid rows

with source as (
    select * from {{ source('geoscience_raw', 'project1_features') }}
),

staged as (
    select
        -- identifiers
        cast(sample_id as string)               as sample_id,

        -- coordinates
        cast(latitude  as double)               as latitude,
        cast(longitude as double)               as longitude,

        -- element concentrations (ppm)
        cast(copper_ppm     as double)          as copper_ppm,
        cast(nickel_ppm     as double)          as nickel_ppm,
        cast(cobalt_ppm     as double)          as cobalt_ppm,
        cast(gold_ppm       as double)          as gold_ppm,
        cast(molybdenum_ppm as double)          as molybdenum_ppm,

        -- anomaly scores
        cast(copper_zscore       as double)     as copper_zscore,
        cast(nickel_zscore       as double)     as nickel_zscore,
        cast(copper_zscore_local as double)     as copper_zscore_local,
        cast(nickel_zscore_local as double)     as nickel_zscore_local,

        -- targeting
        cast(mineralisation_score as double)    as mineralisation_score,
        cast(is_drill_target      as boolean)   as is_drill_target,

        -- geology context
        cast(geology_unit          as string)   as geology_unit,
        cast(dist_to_boundary_km   as double)   as dist_to_boundary_km,

        -- metadata
        current_timestamp()                     as dbt_loaded_at

    from source
    where latitude  is not null
      and longitude is not null
      and copper_ppm > 0
)

select * from staged