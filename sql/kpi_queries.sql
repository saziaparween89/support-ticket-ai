-- ============================================================
-- Support Operations KPI queries (SQLite dialect)
-- Table: tickets  (loaded from data/processed/tickets_clean.csv)
-- Each query is separated by a "-- name:" header so
-- src/sql_analysis.py can run them one by one.
-- ============================================================

-- name: 1. Overall KPI snapshot
SELECT
    COUNT(*)                                             AS total_tickets,
    ROUND(AVG(escalated) * 100, 1)                       AS escalation_rate_pct,
    ROUND(AVG(first_response_min), 1)                    AS avg_first_response_min,
    ROUND(AVG(resolution_hours), 1)                      AS avg_resolution_hrs,
    ROUND(AVG(csat_score), 2)                            AS avg_csat,
    ROUND(100.0 * SUM(CASE WHEN status IN ('Open','Pending') THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                         AS backlog_pct
FROM tickets;

-- name: 2. Category performance (volume, escalation, resolution time, CSAT)
SELECT
    category,
    COUNT(*)                                   AS tickets,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM tickets), 1) AS share_pct,
    ROUND(AVG(escalated) * 100, 1)             AS escalation_rate_pct,
    ROUND(AVG(resolution_hours), 1)            AS avg_resolution_hrs,
    ROUND(AVG(csat_score), 2)                  AS avg_csat
FROM tickets
GROUP BY category
ORDER BY tickets DESC;

-- name: 3. Channel efficiency
SELECT
    channel,
    COUNT(*)                                   AS tickets,
    ROUND(AVG(first_response_min), 1)          AS avg_first_response_min,
    ROUND(AVG(escalated) * 100, 1)             AS escalation_rate_pct,
    ROUND(AVG(csat_score), 2)                  AS avg_csat
FROM tickets
GROUP BY channel
ORDER BY avg_first_response_min;

-- name: 4. Repeat contacts drive escalations
SELECT
    CASE
        WHEN previous_contacts = 0 THEN '0 (first contact)'
        WHEN previous_contacts = 1 THEN '1'
        WHEN previous_contacts = 2 THEN '2'
        ELSE '3+'
    END                                        AS previous_contacts_bucket,
    COUNT(*)                                   AS tickets,
    ROUND(AVG(escalated) * 100, 1)             AS escalation_rate_pct
FROM tickets
GROUP BY previous_contacts_bucket
ORDER BY previous_contacts_bucket;

-- name: 5. Monthly trend (volume + escalation rate)
SELECT
    created_month,
    COUNT(*)                                   AS tickets,
    SUM(escalated)                             AS escalations,
    ROUND(AVG(escalated) * 100, 1)             AS escalation_rate_pct,
    ROUND(AVG(csat_score), 2)                  AS avg_csat
FROM tickets
GROUP BY created_month
ORDER BY created_month;

-- name: 6. Top 10 agents by CSAT (min 150 tickets) - window function ranking
WITH agent_stats AS (
    SELECT
        agent_id,
        COUNT(*)                               AS tickets,
        ROUND(AVG(csat_score), 2)              AS avg_csat,
        ROUND(AVG(resolution_hours), 1)        AS avg_resolution_hrs,
        ROUND(AVG(escalated) * 100, 1)         AS escalation_rate_pct
    FROM tickets
    GROUP BY agent_id
    HAVING COUNT(*) >= 150
)
SELECT
    RANK() OVER (ORDER BY avg_csat DESC)       AS csat_rank,
    *
FROM agent_stats
ORDER BY csat_rank
LIMIT 10;

-- name: 7. Peak hours - when should more agents be rostered?
SELECT
    created_hour,
    COUNT(*)                                   AS tickets,
    ROUND(AVG(escalated) * 100, 1)             AS escalation_rate_pct
FROM tickets
GROUP BY created_hour
ORDER BY tickets DESC
LIMIT 5;

-- name: 8. Premium customers at risk: escalation rate by tier vs overall
SELECT
    customer_tier,
    COUNT(*)                                   AS tickets,
    ROUND(AVG(escalated) * 100, 1)             AS escalation_rate_pct,
    ROUND(AVG(escalated) * 100 - (SELECT AVG(escalated) * 100 FROM tickets), 1)
                                               AS vs_overall_pp
FROM tickets
GROUP BY customer_tier
ORDER BY escalation_rate_pct DESC;
