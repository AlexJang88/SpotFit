-- Disposable test DB only. All fixtures roll back; never an initial seed.
\set ON_ERROR_STOP on
BEGIN;
SET LOCAL search_path=spotfit,public,pg_catalog;
CREATE TEMP TABLE checks(name text PRIMARY KEY);
CREATE FUNCTION pg_temp.expect_error(label text,statement text,expected_state text DEFAULT '23514') RETURNS void LANGUAGE plpgsql AS $$
BEGIN
 BEGIN
  EXECUTE statement;
 EXCEPTION WHEN OTHERS THEN
  IF SQLSTATE<>expected_state THEN RAISE EXCEPTION 'Test %: expected %, got %: %',label,expected_state,SQLSTATE,SQLERRM; END IF;
  INSERT INTO checks VALUES(label); RETURN;
 END;
 RAISE EXCEPTION 'Test %: statement unexpectedly succeeded',label;
END $$;
DO $tests$
DECLARE gu bigint; dong bigint; dong2 bigint; pipeline bigint; dataset bigint; train bigint;
 cm bigint; tm bigint; pm bigint; cp bigint; tp bigint; pp bigint; pp2 bigint; score_id bigint;
 policy bigint; rec_policy bigint; req uuid; batch uuid:=gen_random_uuid(); n bigint;
BEGIN
 SELECT area_id INTO gu FROM ref_admin_area WHERE area_level='DISTRICT' AND area_code='11110' AND is_current;
 SELECT area_id INTO dong FROM ref_admin_area WHERE parent_area_id=gu ORDER BY area_code LIMIT 1;
 SELECT area_id INTO dong2 FROM ref_admin_area WHERE parent_area_id=gu ORDER BY area_code OFFSET 1 LIMIT 1;
 SELECT scoring_policy_id INTO policy FROM service_scoring_policy WHERE policy_code='NEIGHBORHOOD_50_50_V1';
 SELECT recommendation_policy_id INTO rec_policy FROM service_recommendation_policy WHERE policy_code='BUDGET_TOP5_DISTINCT_GU_V1';
 PERFORM pg_temp.expect_error('area overlap',format('INSERT INTO ref_admin_area(area_code,area_name,area_level,valid_from) VALUES (%L,%L,%L,%L)','11','duplicate','CITY','2026-11-01'),'23P01');
 PERFORM pg_temp.expect_error('hierarchy cycle/level',format('UPDATE ref_admin_area SET parent_area_id=%s WHERE area_id=%s',dong,gu));
 PERFORM pg_temp.expect_error('weights sum', $$INSERT INTO service_scoring_policy(policy_code,policy_name,crime_weight,traffic_weight,valid_from) VALUES ('_bad','bad',.7,.5,'2026-11-16')$$);
 PERFORM pg_temp.expect_error('budget range',format('INSERT INTO service_recommendation_request(recommendation_policy_id,transaction_type,minimum_budget,maximum_budget,crime_prediction_year) VALUES (%s,%L,60000,30000,2027)',rec_policy,'SALE'));
 PERFORM pg_temp.expect_error('NaN money',format('INSERT INTO service_recommendation_request(recommendation_policy_id,transaction_type,maximum_budget,crime_prediction_year) VALUES (%s,%L,%L,2027)',rec_policy,'SALE','NaN'));
 PERFORM pg_temp.expect_error('dangling FK', $$INSERT INTO core_population_stat(area_id,asset_version_id,reference_year,age_group,population) VALUES (-999,-999,2025,'0-17',1)$$,'23503');
 INSERT INTO meta_pipeline_run(pipeline_type,status,finished_at) VALUES ('TEST','SUCCEEDED',now()) RETURNING pipeline_run_id INTO pipeline;
 INSERT INTO feature_dataset_version(pipeline_run_id,domain,feature_version,reference_date,storage_uri,schema_hash,row_count,quality_passed)
 VALUES (pipeline,'TEST','fixture','2025-12-31','test://features',repeat('a',64),25,true) RETURNING feature_dataset_id INTO dataset;
 INSERT INTO ml_training_run(feature_dataset_id,pipeline_run_id,model_type,algorithm,status,finished_at)
 VALUES(dataset,pipeline,'CRIME_EXPOSURE','fixture','SUCCEEDED',now()) RETURNING training_run_id INTO train;
 INSERT INTO ml_model_version(training_run_id,model_type,version,prediction_horizon,artifact_uri,model_hash,status,validation_passed)
 VALUES(train,'CRIME_EXPOSURE','_test',2,'test://crime',repeat('b',64),'VALIDATED',true) RETURNING model_version_id INTO cm;
 PERFORM pg_temp.expect_error('deployment gate',format('INSERT INTO ml_model_deployment(model_version_id,environment,status,deployed_at) VALUES (%s,%L,%L,now())',cm,'DEV','ACTIVE'));
 UPDATE ml_model_version SET is_champion=true WHERE model_version_id=cm;
 INSERT INTO ml_model_metric(model_version_id,dataset_split,metric_name,metric_value) VALUES (cm,'CV','Spearman',.97);
 INSERT INTO ml_model_feature(model_version_id,feature_name,weight_pct,direction,display_order) VALUES (cm,'a',60,'HIGHER_RISK',1),(cm,'b',20,'LOWER_RISK',2);
 PERFORM pg_temp.expect_error('feature weight sum gate',format('INSERT INTO ml_model_deployment(model_version_id,environment,status,deployed_at) VALUES (%s,%L,%L,now())',cm,'DEV','ACTIVE'));
 UPDATE ml_model_feature SET weight_pct=40 WHERE model_version_id=cm AND feature_name='b';
 INSERT INTO ml_model_deployment(model_version_id,environment,status,deployed_at) VALUES (cm,'DEV','ACTIVE',now());
 INSERT INTO checks VALUES('validated deployment accepted');
 PERFORM pg_temp.expect_error('used feature freeze',format('UPDATE ml_model_feature SET weight_pct=10 WHERE model_version_id=%s',cm));
 PERFORM pg_temp.expect_error('model identity freeze',format('UPDATE ml_model_version SET prediction_horizon=3 WHERE model_version_id=%s',cm));
 PERFORM pg_temp.expect_error('one champion',format('INSERT INTO ml_model_version(training_run_id,model_type,version,prediction_horizon,artifact_uri,model_hash,status,validation_passed,is_champion) VALUES (%s,%L,%L,2,%L,%L,%L,true,true)',train,'CRIME_EXPOSURE','_test2','test://crime2',repeat('c',64),'VALIDATED'),'23505');
 INSERT INTO ml_training_run(feature_dataset_id,pipeline_run_id,model_type,algorithm,status,finished_at)
 VALUES(dataset,pipeline,'TRAFFIC_ACCIDENT','fixture','SUCCEEDED',now()) RETURNING training_run_id INTO train;
 INSERT INTO ml_model_version(training_run_id,model_type,version,prediction_horizon,artifact_uri,model_hash)
 VALUES(train,'TRAFFIC_ACCIDENT','_test',0,'test://traffic',repeat('d',64)) RETURNING model_version_id INTO tm;
 INSERT INTO ml_training_run(feature_dataset_id,pipeline_run_id,model_type,algorithm,status,finished_at)
 VALUES(dataset,pipeline,'APARTMENT_SALE','fixture','SUCCEEDED',now()) RETURNING training_run_id INTO train;
 INSERT INTO ml_model_version(training_run_id,model_type,version,prediction_horizon,artifact_uri,model_hash)
 VALUES(train,'APARTMENT_SALE','_test',0,'test://property',repeat('e',64)) RETURNING model_version_id INTO pm;
 PERFORM pg_temp.expect_error('horizon mismatch',format('INSERT INTO service_safety_prediction(model_version_id,area_id,safety_domain,reference_year,prediction_year,predicted_value,safety_score) VALUES (%s,%s,%L,2025,2028,7,92.6)',cm,gu,'CRIME_EXPOSURE'));
 PERFORM pg_temp.expect_error('safety is gu not dong',format('INSERT INTO service_safety_prediction(model_version_id,area_id,safety_domain,reference_year,prediction_year,predicted_value,safety_score) VALUES (%s,%s,%L,2025,2027,7,92.6)',cm,dong,'CRIME_EXPOSURE'));
 INSERT INTO service_safety_prediction(model_version_id,area_id,safety_domain,reference_year,prediction_year,predicted_value,safety_score)
 VALUES(cm,gu,'CRIME_EXPOSURE',2025,2027,7,92.6) RETURNING safety_prediction_id INTO cp;
 INSERT INTO service_safety_prediction(model_version_id,area_id,safety_domain,reference_year,prediction_year,prediction_kind,predicted_value,safety_score)
 VALUES(tm,gu,'TRAFFIC_ACCIDENT',2025,2025,'REFERENCE',1,100) RETURNING safety_prediction_id INTO tp;
 PERFORM pg_temp.expect_error('bad weighted score',format('INSERT INTO service_neighborhood_score(score_batch_key,area_id,scoring_policy_id,crime_prediction_id,traffic_prediction_id,reference_year,prediction_year,crime_score,traffic_score,total_score,total_rank,total_grade) VALUES (%L,%s,%s,%s,%s,2025,2027,92.6,100,100,1,%L)',batch,gu,policy,cp,tp,'A'));
 INSERT INTO service_neighborhood_score(score_batch_key,area_id,scoring_policy_id,crime_prediction_id,traffic_prediction_id,reference_year,prediction_year,crime_score,traffic_score,total_score,total_rank,total_grade)
 VALUES(batch,gu,policy,cp,tp,2025,2027,92.6,100,96.3,1,'A') RETURNING neighborhood_score_id INTO score_id;
 INSERT INTO checks VALUES('crime 2027 + traffic reference 2025 accepted');
 PERFORM pg_temp.expect_error('immutable prediction',format('UPDATE service_safety_prediction SET safety_score=0 WHERE safety_prediction_id=%s',cp));
 INSERT INTO service_property_prediction(model_version_id,area_id,transaction_type,reference_date,prediction_date,prediction_kind,predicted_amount,predicted_price_per_sqm)
 VALUES(pm,dong,'SALE','2026-09-30','2026-09-30','ESTIMATE',50000,500) RETURNING property_prediction_id INTO pp;
 PERFORM pg_temp.expect_error('nullable prediction key unique',format('INSERT INTO service_property_prediction(model_version_id,area_id,transaction_type,reference_date,prediction_date,prediction_kind,predicted_amount,predicted_price_per_sqm) VALUES (%s,%s,%L,%L,%L,%L,50000,500)',pm,dong,'SALE','2026-09-30','2026-09-30','ESTIMATE'),'23505');
 INSERT INTO service_property_prediction(model_version_id,area_id,transaction_type,reference_date,prediction_date,prediction_kind,predicted_amount,predicted_price_per_sqm)
 VALUES(pm,dong2,'SALE','2026-09-30','2026-09-30','ESTIMATE',50000,500) RETURNING property_prediction_id INTO pp2;
 INSERT INTO service_recommendation_request(recommendation_policy_id,transaction_type,minimum_budget,maximum_budget,crime_prediction_year)
 VALUES(rec_policy,'SALE',30000,60000,2027) RETURNING request_id INTO req;
 PERFORM pg_temp.expect_error('recommendation wrong amount',format('INSERT INTO service_recommendation_result(request_id,recommendation_rank,area_id,property_prediction_id,neighborhood_score_id,predicted_amount,neighborhood_score,neighborhood_grade) VALUES (%L,1,%s,%s,%s,70000,96.3,%L)',req,dong,pp,score_id,'A'));
 INSERT INTO service_recommendation_result(request_id,recommendation_rank,area_id,property_prediction_id,neighborhood_score_id,predicted_amount,neighborhood_score,neighborhood_grade)
 VALUES(req,1,dong,pp,score_id,50000,96.3,'A');
 INSERT INTO checks VALUES('valid recommendation accepted');
 PERFORM pg_temp.expect_error('one dong per district',format('INSERT INTO service_recommendation_result(request_id,recommendation_rank,area_id,property_prediction_id,neighborhood_score_id,predicted_amount,neighborhood_score,neighborhood_grade) VALUES (%L,2,%s,%s,%s,50000,96.3,%L)',req,dong2,pp2,score_id,'A'));
 INSERT INTO analytics_service_event(request_id,event_name) VALUES(req,'SUBMIT_RECOMMENDATION');
 DELETE FROM service_recommendation_request WHERE request_id=req;
 IF EXISTS(SELECT 1 FROM service_recommendation_result WHERE request_id=req) THEN RAISE EXCEPTION 'Request cascade failed'; END IF;
 IF NOT EXISTS(SELECT 1 FROM analytics_service_event WHERE request_id IS NULL AND event_name='SUBMIT_RECOMMENDATION') THEN RAISE EXCEPTION 'Event retention failed'; END IF;
 INSERT INTO checks VALUES('request retention cascade/set-null accepted');
END $tests$;
SELECT name,'PASS' AS status FROM checks ORDER BY name;
SELECT count(*) AS passed_checks FROM checks;
ROLLBACK;
