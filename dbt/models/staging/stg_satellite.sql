-- stg_satellite.sql
-- Staging model for Project 2 satellite spectral features
-- Standardises column names and filters invalid alteration scores

with source as (
    select * from {{ source('geoscience_raw', 'project2_satellite') }}
),

staged as (
    select
        -- coordinates
        cast(easting  as double)                as easting,
        cast(northing as double)                as northing,

        -- spectral indices
        cast(ndvi             as double)        as ndvi,
        cast(iron_oxide_ratio as double)        as iron_oxide_ratio,
        cast(clay_ratio       as double)        as clay_ratio,
        cast(ferrous_index    as double)        as ferrous_index,

        -- alteration score
        cast(alteration_score    as double)     as alteration_score,
        cast(is_high_alteration  as boolean)    as is_high_alteration,

        -- scene metadata
        cast(scene_id as string)                as scene_id,
        cast(sensor   as string)                as sensor,

        -- metadata
        current_timestamp()                     as dbt_loaded_at

    from source
    where alteration_score is not null
      and easting           is not null
      and northing          is not null
)

select * from staged