-- int_drill_targets.sql
-- Identifies high-value drill intervals
-- High grade OR in mineralised zone with good recovery

with staged as (
    select * from {{ ref('stg_drilling') }}
),

targets as (
    select
        *,
        -- grade category
        case
            when copper_ppm >= 1000 then 'BONANZA'
            when copper_ppm >= 500  then 'HIGH_GRADE'
            when copper_ppm >= 200  then 'ECONOMIC'
            when copper_ppm >= 50   then 'SUB_ECONOMIC'
            else 'BACKGROUND'
        end                                     as grade_category,

        -- value score combining grade and depth context
        case
            when in_mineralised_zone = true
             and copper_ppm >= 200
            then copper_ppm * 1.5
            else copper_ppm
        end                                     as weighted_grade

    from staged
),

hole_summary as (
    select
        hole_id,
        count(*)                                as n_composites,
        max(copper_ppm)                         as max_cu_ppm,
        avg(copper_ppm)                         as avg_cu_ppm,
        sum(case when in_mineralised_zone
            then interval_length else 0 end)    as mineralised_thickness_m,
        max(weighted_grade)                     as best_weighted_grade
    from targets
    group by hole_id
)

select
    t.*,
    h.max_cu_ppm,
    h.avg_cu_ppm,
    h.mineralised_thickness_m,
    h.best_weighted_grade
from targets t
join hole_summary h
    on t.hole_id = h.hole_id
where t.is_high_grade = true
   or t.in_mineralised_zone = true