-- stg_drilling.sql
-- Staging model for Project 3 drill hole composites
-- Standardises units and filters poor recovery intervals

with source as (
    select * from {{ source('geoscience_raw', 'project3_composites') }}
),

staged as (
    select
        -- identifiers
        cast(HoleID as string)                  as hole_id,

        -- depth
        cast(From       as double)              as depth_from,
        cast(To         as double)              as depth_to,
        cast(MidDepth   as double)              as mid_depth,
        cast(Length     as double)              as interval_length,

        -- 3D coordinates
        cast(X as double)                       as easting,
        cast(Y as double)                       as northing,
        cast(Z as double)                       as elevation,

        -- grade (ppm)
        cast(Cu_ppm  as double)                 as copper_ppm,
        cast(Ni_ppm  as double)                 as nickel_ppm,
        cast(Co_ppm  as double)                 as cobalt_ppm,
        cast(Au_ppb  as double)                 as gold_ppb,
        cast(Mo_ppm  as double)                 as molybdenum_ppm,

        -- quality
        cast(Recovery_pct as double)            as recovery_pct,
        cast(n_samples    as integer)           as n_samples,

        -- derived flags
        case
            when cast(Cu_ppm as double) >= 200  then true
            else false
        end                                     as is_high_grade,

        case
            when cast(MidDepth as double) between 80 and 150 then true
            else false
        end                                     as in_mineralised_zone,

        -- metadata
        current_timestamp()                     as dbt_loaded_at

    from source
    where HoleID    is not null
      and Cu_ppm    is not null
      and Recovery_pct > 50     -- exclude very poor recovery intervals
)

select * from staged