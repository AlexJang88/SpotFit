-- Run inside the installer transaction, or independently with psql -v ON_ERROR_STOP=1.
DO $$
DECLARE n integer;
BEGIN
 SELECT count(*) INTO n FROM pg_tables WHERE schemaname='spotfit';
 IF n<35 THEN RAISE EXCEPTION 'Expected at least 35 tables (33 ERD + model catalog + migration ledger), got %',n; END IF;
 IF (SELECT count(*) FROM spotfit.ml_model_type)<>5 THEN RAISE EXCEPTION 'Expected five model types'; END IF;
 IF (SELECT count(*) FROM spotfit.ref_admin_area WHERE code_system='MOIS' AND valid_from='2026-10-01' AND area_level='CITY')<>1
 OR (SELECT count(*) FROM spotfit.ref_admin_area WHERE code_system='MOIS' AND valid_from='2026-10-01' AND area_level='DISTRICT')<>25
 OR (SELECT count(*) FROM spotfit.ref_admin_area WHERE code_system='MOIS' AND valid_from='2026-10-01' AND area_level='DONG')<>427 THEN
  RAISE EXCEPTION 'Expected the complete 2026-10 snapshot: 1 city / 25 districts / 427 dongs';
 END IF;
 IF EXISTS (SELECT 1 FROM spotfit.ref_admin_area WHERE valid_from='2026-10-01' AND code_system='MOIS'
   AND (boundary IS NULL OR NOT ST_IsValid(boundary) OR ST_IsEmpty(boundary) OR ST_SRID(boundary)<>4326)) THEN
  RAISE EXCEPTION 'Missing or invalid initial boundary geometry';
 END IF;
 IF NOT EXISTS (SELECT 1 FROM spotfit.service_scoring_policy WHERE policy_code='NEIGHBORHOOD_50_50_V1' AND crime_weight=0.5 AND traffic_weight=0.5) THEN
  RAISE EXCEPTION 'Baseline scoring policy missing or differs from the seed';
 END IF;
 IF (SELECT count(*) FROM spotfit.service_grade_policy g JOIN spotfit.service_scoring_policy p USING(scoring_policy_id)
     WHERE p.policy_code='NEIGHBORHOOD_50_50_V1' AND maximum_rank-minimum_rank=4)<>5 THEN
  RAISE EXCEPTION 'Expected A-E, each covering five ranks';
 END IF;
 IF NOT EXISTS (SELECT 1 FROM spotfit.service_recommendation_policy WHERE policy_code='BUDGET_TOP5_DISTINCT_GU_V1' AND recommendation_count=5 AND one_dong_per_district) THEN
  RAISE EXCEPTION 'Baseline recommendation policy missing';
 END IF;
END $$;
SELECT area_level,count(*) AS seeded_regions FROM spotfit.ref_admin_area
WHERE valid_from='2026-10-01' AND code_system='MOIS' GROUP BY area_level ORDER BY area_level;
SELECT model_type,display_name FROM spotfit.ml_model_type ORDER BY model_type;
