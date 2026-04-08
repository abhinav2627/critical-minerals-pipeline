-- mart_unified_targets.sql
-- GOLD LAYER: Final unified exploration targeting table
-- Combines ML predictions with geochemistry anomaly context
-- This is the primary output table for the entire pipeline

with ml as (
    select * from {{ ref('stg_ml_predictions') }}
),

geochem as (
    select * from {{ ref('int_geochemistry_anomalies') }}
),

final as (
    select
        -- location
        ml.latitude,
        ml.longitude,

        -- ML targeting
        ml.deposit_probability,
        ml.ml_rank,
        ml.probability_tier,
        ml.is_high_probability,

        -- geochemistry context
        ml.copper_zscore_local,
        ml.nickel_zscore_local,
        ml.mineralisation_score,
        

        -- geophysics context
        ml.geo4_mag_anomaly_nt,
        ml.geo4_mag_residual_nt,
        ml.geo4_pseudogravity,

        -- convergence scoring
        -- HIGH probability + significant geochem anomaly = top target
        case
            when ml.deposit_probability >= 0.8
             and ml.mineralisation_score >= 1.0
            then 'CONVERGENCE_TARGET'
            when ml.deposit_probability >= 0.7
            then 'ML_TARGET'
            when ml.mineralisation_score >= 2.0
            then 'GEOCHEM_TARGET'
            else 'BACKGROUND'
        end                                     as target_classification,

        -- final priority rank
        row_number() over (
            order by ml.deposit_probability desc,
                     ml.mineralisation_score  desc
        )                                       as priority_rank,

        -- metadata
        ml.dbt_loaded_at

    from ml
)

select * from final
order by priority_rank