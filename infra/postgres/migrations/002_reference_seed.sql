-- Real reference/configuration data only. No dummy training or prediction rows.
INSERT INTO spotfit.ml_model_type(model_type,display_name,domain,target_description) VALUES
 ('CRIME_EXPOSURE','아동 지역범죄 노출 모델','SAFETY','log(1 + 5대범죄 / 0~17세 인구 * 10000); 실제 아동 피해율이 아닌 대리지표'),
 ('TRAFFIC_ACCIDENT','어린이 교통사고·교통안전 모델','SAFETY','자치구 교통사고 위험·안전 수준'),
 ('APARTMENT_SALE','아파트 매매가격 모델','PROPERTY','매매 총액 또는 ㎡당 금액; 단위 만원'),
 ('APARTMENT_JEONSE','아파트 전세가격 모델','PROPERTY','전세 보증금 또는 ㎡당 보증금; 단위 만원'),
 ('APARTMENT_MONTHLY_RENT','아파트 전월세가격 모델','PROPERTY','보증금·면적·건물 조건에 따른 월세; 단위 만원')
ON CONFLICT (model_type) DO NOTHING;

-- URLs identify the portal/catalog; not all rows are machine-download endpoints.
INSERT INTO spotfit.meta_data_source(source_code,source_name,provider,source_url,update_cycle) VALUES
 ('SEOUL_CRIME','서울시 5대범죄 발생 통계','서울 열린데이터광장','https://data.seoul.go.kr/','ANNUAL_SOURCE_CONFIRM'),
 ('MOIS_AGE_POPULATION','연령별 주민등록인구','행정안전부','https://jumin.mois.go.kr/ageStatMonth.do','MONTHLY'),
 ('SEOUL_TRAFFIC','서울시 교통사고 통계 자료','서울 열린데이터광장','https://data.seoul.go.kr/','SOURCE_DEPENDENT'),
 ('APARTMENT_TRANSACTIONS','아파트 매매·전월세 실거래 자료','국토교통부 실거래가 공개시스템','https://rt.molit.go.kr/','SOURCE_DEPENDENT'),
 ('MOIS_BOUNDARY_NOTICE','행정구역·주민등록주소코드 변경 공고','행정안전부','https://www.mois.go.kr/frt/bbs/type001/commonSelectBoardList.do?bbsId=BBSMSTR_000000000052','EVENT_BASED'),
 ('SGIS_BOUNDARY','행정동 경계·코드 스냅샷','SGIS / vuski/admdongkor','https://sgis.kostat.go.kr/developer/html/openApi/api/data.html','SNAPSHOT_2026_10_01')
ON CONFLICT (source_code) DO NOTHING;

INSERT INTO spotfit.service_scoring_policy(policy_code,policy_name,crime_weight,traffic_weight,valid_from,is_active)
VALUES ('NEIGHBORHOOD_50_50_V1','범죄 Feature 가중합 50% + 교통안전 50%',0.5,0.5,'2026-11-16',true)
ON CONFLICT DO NOTHING;
INSERT INTO spotfit.service_grade_policy(scoring_policy_id,grade,minimum_rank,maximum_rank,color,fill_opacity,display_order)
SELECT p.scoring_policy_id,g.grade,g.lo,g.hi,'#526e68',g.opacity,g.ord
FROM spotfit.service_scoring_policy p CROSS JOIN (VALUES
 ('A',1,5,0.75,1),('B',6,10,0.58,2),('C',11,15,0.42,3),('D',16,20,0.26,4),('E',21,25,0.12,5)
) g(grade,lo,hi,opacity,ord)
WHERE p.policy_code='NEIGHBORHOOD_50_50_V1'
ON CONFLICT DO NOTHING;
INSERT INTO spotfit.service_recommendation_policy(policy_code,policy_name,recommendation_count,one_dong_per_district,exclude_missing_price,sort_rule,is_active)
VALUES ('BUDGET_TOP5_DISTINCT_GU_V1','예산 내 구별 최저가 행정동 TOP 5',5,true,true,
 '예산 범위 내 구별 최저가 행정동 1곳 → 동네점수 DESC → 범죄안전 점수 DESC → 자치구명 ASC',true)
ON CONFLICT DO NOTHING;
