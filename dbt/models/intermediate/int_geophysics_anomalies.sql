-- int_geophysics_anomalies.sql
-- Filters geophysics grid to magnetic anomaly zones only
-- High residual anomaly indicates shallow magnetite-bearing sources

with staged as (
    select * from {{ ref('stg_geophysics') }}
),

anomalies as (
    select
        *,
        -- anomaly strength category
        case
            when mag_zscore >= 3.0  then 'STRONG'
            when mag_zscore >= 2.0  then 'MODERATE'
            when mag_zscore >= 1.0  then 'WEAK'
            else 'BACKGROUND'
        end                                     as anomaly_strength,

        -- pseudogravity category
        case
            when pseudogravity >= 1.0   then 'DENSE_SOURCE'
            when pseudogravity >= 0.0   then 'NEUTRAL'
            else 'LIGHT_SOURCE'
        end                                     as gravity_signature

    from staged
    where is_mag_high = true
       or mag_zscore  > 1.5
)

select * from anomalies