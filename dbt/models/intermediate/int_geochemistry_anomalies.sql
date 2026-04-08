-- int_geochemistry_anomalies.sql
-- Filters geochemistry staging to significant anomalies only
-- Z-score > 1.5 in any pathfinder element

with staged as (
    select * from {{ ref('stg_geochemistry') }}
),

anomalies as (
    select
        *,
        -- composite anomaly flag
        case
            when copper_zscore_local > 2.0 then true
            when nickel_zscore_local > 2.0 then true
            else false
        end                                     as is_significant_anomaly,

        -- anomaly tier
        case
            when mineralisation_score >= 4.0    then 'TIER_1'
            when mineralisation_score >= 2.0    then 'TIER_2'
            when mineralisation_score >= 1.0    then 'TIER_3'
            else 'BACKGROUND'
        end                                     as anomaly_tier,

        -- pathfinder ratio
        case
            when nickel_ppm > 0
            then copper_ppm / nickel_ppm
            else null
        end                                     as cu_ni_ratio

    from staged
)

select * from anomalies
where is_significant_anomaly = true
   or mineralisation_score   > 1.0