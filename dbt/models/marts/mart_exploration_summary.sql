-- mart_exploration_summary.sql
-- GOLD LAYER: Summary analytics by probability tier
-- Aggregates all targeting signals for executive reporting

with targets as (
    select * from {{ ref('mart_unified_targets') }}
),

summary as (
    select
        probability_tier,
        target_classification,

        -- counts
        count(*)                                            as total_locations,
        sum(case when target_classification = 'CONVERGENCE_TARGET'
            then 1 else 0 end)                             as convergence_targets,
        sum(case when target_classification = 'ML_TARGET'
            then 1 else 0 end)                             as ml_targets,
        sum(case when target_classification = 'GEOCHEM_TARGET'
            then 1 else 0 end)                             as geochem_targets,

        -- probability stats
        round(avg(deposit_probability), 4)                 as avg_deposit_probability,
        round(max(deposit_probability), 4)                 as max_deposit_probability,
        sum(case when is_high_probability = true
            then 1 else 0 end)                             as high_probability_count,

        -- geochemistry stats
        round(avg(mineralisation_score), 3)                as avg_mineralisation_score,
        round(max(mineralisation_score), 3)                as max_mineralisation_score,

        -- geophysics stats
        round(avg(geo4_mag_anomaly_nt), 2)                 as avg_mag_anomaly_nt,
        round(avg(geo4_pseudogravity),  4)                 as avg_pseudogravity

    from targets
    group by probability_tier, target_classification
)

select
    *,
    round(
        convergence_targets * 1.0 / nullif(total_locations, 0),
        4
    )                                                       as convergence_rate
from summary
order by avg_deposit_probability desc