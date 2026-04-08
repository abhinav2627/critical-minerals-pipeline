-- stg_geophysics.sql
-- Staging model for Project 4 airborne geophysics grid
-- GeoDAWN USGS EarthMRI 2022 aeromagnetic survey

with source as (
    select * from {{ source('geoscience_raw', 'project4_geophysics') }}
),

staged as (
    select
        -- coordinates
        cast(latitude  as double)               as latitude,
        cast(longitude as double)               as longitude,

        -- magnetic features
        cast(mag_anomaly_nT  as double)         as mag_anomaly_nt,
        cast(mag_residual_nT as double)         as mag_residual_nt,
        cast(pseudogravity   as double)         as pseudogravity,
        cast(mag_zscore      as double)         as mag_zscore,

        -- anomaly flag
        cast(is_mag_high as boolean)            as is_mag_high,

        -- metadata
        current_timestamp()                     as dbt_loaded_at

    from source
    where latitude  is not null
      and longitude is not null
      and mag_anomaly_nT is not null
)

select * from staged
