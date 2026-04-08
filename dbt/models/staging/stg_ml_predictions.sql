with source as (
    select * from {{ source('geoscience_raw', 'project5_ml') }}
),

staged as (
    select
        -- coordinates
        cast(latitude  as double)               as latitude,
        cast(longitude as double)               as longitude,

        -- ML outputs
        cast(deposit_probability as double)     as deposit_probability,
        cast(ml_prediction       as integer)    as ml_prediction,
        cast(ml_rank             as integer)    as ml_rank,

        -- input features
        cast(copper_zscore_local  as double)    as copper_zscore_local,
        cast(nickel_zscore_local  as double)    as nickel_zscore_local,
        cast(geo4_mag_anomaly_nT  as double)    as geo4_mag_anomaly_nt,
        cast(geo4_mag_residual_nT as double)    as geo4_mag_residual_nt,
        cast(geo4_pseudogravity   as double)    as geo4_pseudogravity,
        cast(geo4_mag_zscore      as double)    as geo4_mag_zscore,

        -- surface context
        cast(mineralisation_score as double)    as mineralisation_score,

        -- target flags
        case
            when cast(deposit_probability as double) >= 0.8 then true
            else false
        end                                     as is_high_probability,

        case
            when cast(deposit_probability as double) >= 0.9 then 'HIGH'
            when cast(deposit_probability as double) >= 0.7 then 'MEDIUM'
            when cast(deposit_probability as double) >= 0.5 then 'LOW'
            else 'BACKGROUND'
        end                                     as probability_tier,

        -- metadata
        current_timestamp()                     as dbt_loaded_at

    from source
    where latitude            is not null
      and longitude           is not null
      and deposit_probability is not null
)

select * from staged