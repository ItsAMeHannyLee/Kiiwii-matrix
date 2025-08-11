from pytrends.request import TrendReq
from PublicDataReader import Kosis
from pandas import json_normalize
import country_converter as coco
import tradingeconomics as te
import comtradeapicall as cac
import pandas as pd
import requests
import sqlite3
import random
import os

te.login('660A8C420E65485:BB631AE5B0AB4E2')

# API KEYS
UNCOMTRADE_SUBKEYS = ["3424b551bf0c4fddb9abbf1843272807", 
					  "6312a3aed6034aebb217e0df367c7488", 
					  "930ce59452aa4ed0b48b26261306c0f9", 
					  "57eaeada7d2a47cb9bf5d0fa8171d726", 
					  "0a2dab2436f84c7ab08cfaa9b7858da8", 
					  "68a2f25c40c84717b597935693c9bd0c", 
					  "c1fcddd86c9d4f22bcd3b82e19d5fdfd"]
KOSIS_API_KEY = 'MWU1ZTYzNjYwMjJmODFkYjg3Nzk1YmM3ZTZjMDBiNzI=' #https://kosis.kr/openapi/index.jsp
api = Kosis(KOSIS_API_KEY)

def collect_TE():
	# read te_countreis.json file
	# countries = pd.read_json('data/te_countries.json', encoding='utf-8')
	# countries = countries['Country'].tolist()
	countries = [
		'United States', 'Japan', 'Vietnam', 'Philippines',
		'Thailand', 'Indonesia', 'Malaysia', 'Singapore',
		'Cambodia', 'Hong Kong', 'Taiwan', 'Australia',
		'New Zealand', 'Canada', 'Mexico', 'Brazil',
		'Argentina', 'Chile', 'India', 'South Korea'
	]

	data = te.getHistoricalData(
		country=countries, 
		indicator=[
			'Consumer Price Index CPI',
			'GDP',
			'PPI Ex Food Energy and Trade Services',
			'Import Prices',
			'Export Prices',
			'interest rate'
			], 
		initDate='2000-01-01',
		output_type='df')
	data.to_csv('data/raw_data/te_data.csv', encoding='utf-8')

def collect_TEV(startYear, endYear, hsCode, flowCode="X", reporterCountry=410, partnerCountry=None, db_path=None):
	"""-------------------------------------------------------------------
	1. 수출입 월간 통계 데이터

	Source: 
	UN Comtrade API = https://uncomtrade.org/docs/list-of-references-parameter-codes/ 	
	DATA.GO.KR 		= https://www.data.go.kr/index.do 									[창민이가 API Key 발급 받는중...]
	"""
	# Set working directory to script location
	os.chdir(os.path.dirname(__file__))

	# Input Parsing
	periods = [
		",".join(f"{year}{month:02d}" for month in range(1, 13))
		for year in range(startYear, endYear + 1)
	]

    # Use script directory for db_path if not provided
	if db_path is None:
		db_path = os.path.join(os.path.dirname(__file__), "data/raw_data", "TEV.db")

    # Call uncomtrade API
	all_data = []
	exclude_codes = [975, 97, 490, 0]

	for i, period in enumerate(periods, 1):
		print(f"Processing period {period[:4]} ({i}/{len(periods)}) for hs_code {hsCode}")
		data = cac.getFinalData(
        UNCOMTRADE_SUBKEYS[1],
        typeCode="C",
        freqCode="M",
        clCode="HS",
        period=period,
        reporterCode=reporterCountry,
        cmdCode=hsCode,
        flowCode=flowCode,
        partnerCode=partnerCountry,
        partner2Code=None,
        customsCode=None,
        motCode=None,
        maxRecords=100000,
        format_output="JSON",
        aggregateBy=None,
        breakdownMode="classic",
        countOnly=None,
        includeDesc=True)

		df = pd.DataFrame(data)
		print(df.head(5))

		try:
			df = df[~df['partnerCode'].isin(exclude_codes)]
		except KeyError:
			print(f"Warning: 'partnerCode' not found in data for period {period} and hs_code {hsCode}.")
        
		try:
			df = df[[
					"period",  "reporterCode", "reporterDesc", "partnerCode",
					"partnerDesc",  "cmdCode",      "cmdDesc",    "flowCode",
					"primaryValue"
				]]
		except KeyError as e:
			print(f"Warning: Missing expected columns in data for period {period} and hs_code {hsCode}. Error: {e}")
			continue

		all_data.append(df)

    # Concatenate all data
	df_final = pd.concat(all_data, ignore_index=True)

    # Ensure data directory exists
	os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)

    # Save to SQLite database only if data exists
	if not df_final.empty:
		conn = sqlite3.connect(db_path)
		df_final.to_sql(f'trade_data', conn, if_exists='append', index=False)
		conn.close()
		print(f"Data for {reporterCountry}, {partnerCountry}, {hsCode}, has been saved.")
	else:
		print(f"No data (excluding 'World' and 'Other' partners) for {reporterCountry}, {partnerCountry}, {hsCode}")

def collect_CPI(startYear, endYear):
	"""-------------------------------------------------------------------
	소비자 물가지수 (CPI)

	Source: 
	KOSIS = https://kosis.kr/statHtml/statHtml.do?list_id=R_SUB_UTITLE_K&obj_var_id=&seqNo=&tblId=DT_2KAA601_02&vw_cd=MT_RTITLE&orgId=101&path=%252FstatisticsList%252FstatisticsListIndex.do&conn_path=MT_RTITLE&itm_id=&lang_mode=ko&scrId=
	API 설명글 = https://wooiljeong.github.io/python/pdr-kosis/
	API 변수 서치 tool = https://kosis.kr/openapi/devGuide/devGuide_0203List.do

	ㅁ 통계 상수
		목록명: 소비자 물가지수
		기관코드: 101
		통계표ID: DT_2KAA601_02
	"""
	# Check if startYear and endYear are valid
	if not (isinstance(startYear, str) and isinstance(endYear, str)):
		startYear = str(startYear)
		endYear = str(endYear)

	df = api.get_data(
	    "통계자료",
	    orgId = "101",
	    tblId = "DT_2KAA601_02",
	    itmId = "ALL",
	    objL1 = "ALL",
	    prdSe = "Y",
	    startPrdDe = startYear,
	    endPrdDe = endYear,
	    )

	df = pd.DataFrame(df)
	df.to_csv('data/raw_data/CPI.csv', encoding='utf-8')
	return df

def collect_PPI(startYear, endYear):
	"""-------------------------------------------------------------------
	생산자 물가지수 (PPI)

	Source: 
	KOSIS = https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3DR_SUB_UTITLE_K%26obj_var_id%3D%26seqNo%3D%26tblId%3DDT_2KAA601_02%26vw_cd%3DMT_RTITLE%26orgId%3D101%26path%3D%252FstatisticsList%252FstatisticsListIndex.do%26conn_path%3DMT_RTITLE%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26
	API 설명글 = https://wooiljeong.github.io/python/pdr-kosis/
	API 변수 서치 tool = https://kosis.kr/openapi/devGuide/devGuide_0203List.do

	ㅁ 통계 상수
		목록명: 생산자 물가지수
		기관코드: 101
		통계표ID: DT_2KAA601_01
	"""
	# Check if startYear and endYear are valid
	if not (isinstance(startYear, str) and isinstance(endYear, str)):
		startYear = str(startYear)
		endYear = str(endYear)

	df = api.get_data(
	    "통계자료",
	    orgId = "101",
	    tblId = "DT_2KAA601_01",
	    itmId = "ALL",
	    objL1 = "ALL",
	    prdSe = "Y",
	    startPrdDe = startYear,
	    endPrdDe = endYear,
	    )

	df = pd.DataFrame(df)
	df.to_csv('data/raw_data/PPI.csv', encoding='utf-8')
	return df

def collect_IR(startYear, endYear):
	"""-------------------------------------------------------------------
	중앙은행 기준금리

	Source: 
	KOSIS = https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3DR_SUB_UTITLE_K%26obj_var_id%3D%26seqNo%3D%26tblId%3DDT_2KAA601_02%26vw_cd%3DMT_RTITLE%26orgId%3D101%26path%3D%252FstatisticsList%252FstatisticsListIndex.do%26conn_path%3DMT_RTITLE%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26
	API 설명글 = https://wooiljeong.github.io/python/pdr-kosis/
	API 변수 서치 tool = https://kosis.kr/openapi/devGuide/devGuide_0203List.do

	ㅁ 통계 상수
		목록명: 중앙은행 기준금리
		기관코드: 101
		통계표ID: DT_2OEEO032
	"""
	# Check if startYear and endYear are valid
	if not (isinstance(startYear, str) and isinstance(endYear, str)):
		startYear = str(startYear)
		endYear = str(endYear)

	df = api.get_data(
	    "통계자료",
	    orgId = "101",
	    tblId = "DT_2OEEO032",
	    itmId = "ALL",
	    objL1 = "ALL",
	    prdSe = "Y",
	    startPrdDe = startYear,
	    endPrdDe = endYear,
	    )

	df = pd.DataFrame(df)
	df.to_csv('data/raw_data/IR.csv', encoding='utf-8')
	return df

def collect_ER(startYear, endYear):
	"""-------------------------------------------------------------------
	환율

	Source: 
	KOSIS = https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3DR_SUB_UTITLE_K%26obj_var_id%3D%26seqNo%3D%26tblId%3DDT_2KAA601_02%26vw_cd%3DMT_RTITLE%26orgId%3D101%26path%3D%252FstatisticsList%252FstatisticsListIndex.do%26conn_path%3DMT_RTITLE%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26
	API 설명글 = https://wooiljeong.github.io/python/pdr-kosis/
	API 변수 서치 tool = https://kosis.kr/openapi/devGuide/devGuide_0203List.do

	ㅁ 통계 상수
		목록명: 환율
		기관코드: 101
		통계표ID: DT_2KAA811
	"""
	# Check if startYear and endYear are valid
	if not (isinstance(startYear, str) and isinstance(endYear, str)):
		startYear = str(startYear)
		endYear = str(endYear)

	df = api.get_data(
	    "통계자료",
	    orgId = "101",
	    tblId = "DT_2KAA811",
	    itmId = "ALL",
	    objL1 = "ALL",
	    prdSe = "Y",
	    startPrdDe = startYear,
	    endPrdDe = endYear,
	    )

	df = pd.DataFrame(df)
	df.to_csv('data/raw_data/ER.csv', encoding='utf-8')
	return df

def collect_EAI(startYear, endYear):
	"""-------------------------------------------------------------------
	국가별 수출 물량지수

	Source: 
	KOSIS = https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3DR_SUB_UTITLE_K%26obj_var_id%3D%26seqNo%3D%26tblId%3DDT_2KAA601_02%26vw_cd%3DMT_RTITLE%26orgId%3D101%26path%3D%252FstatisticsList%252FstatisticsListIndex.do%26conn_path%3DMT_RTITLE%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26
	API 설명글 = https://wooiljeong.github.io/python/pdr-kosis/
	API 변수 서치 tool = https://kosis.kr/openapi/devGuide/devGuide_0203List.do

	ㅁ 통계 상수
		목록명: 수출·수입 물량지수
		기관코드: 101
		통계표ID: DT_2KAA807
	"""
	# Check if startYear and endYear are valid
	if not (isinstance(startYear, str) and isinstance(endYear, str)):
		startYear = str(startYear)
		endYear = str(endYear)

	df = api.get_data(
	    "통계자료",
	    orgId = "101",
	    tblId = "DT_2KAA807",
	    itmId = "T10",
	    objL1 = "ALL",
	    prdSe = "Y",
	    startPrdDe = startYear,
	    endPrdDe = endYear,
	    )

	df = pd.DataFrame(df)
	df.to_csv('data/raw_data/EAI.csv', encoding='utf-8')
	return df

def collect_IAI(startYear, endYear):
	"""-------------------------------------------------------------------
	국가별 수입 물량지수

	Source: 
	KOSIS = https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3DR_SUB_UTITLE_K%26obj_var_id%3D%26seqNo%3D%26tblId%3DDT_2KAA601_02%26vw_cd%3DMT_RTITLE%26orgId%3D101%26path%3D%252FstatisticsList%252FstatisticsListIndex.do%26conn_path%3DMT_RTITLE%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26
	API 설명글 = https://wooiljeong.github.io/python/pdr-kosis/
	API 변수 서치 tool = https://kosis.kr/openapi/devGuide/devGuide_0203List.do

	ㅁ 통계 상수
		목록명: 수출·수입 물량지수
		기관코드: 101
		통계표ID: DT_2KAA807
	"""
	# Check if startYear and endYear are valid
	if not (isinstance(startYear, str) and isinstance(endYear, str)):
		startYear = str(startYear)
		endYear = str(endYear)

	df = api.get_data(
	    "통계자료",
	    orgId = "101",
	    tblId = "DT_2KAA807",
	    itmId = "T20",
	    objL1 = "ALL",
	    prdSe = "Y",
	    startPrdDe = startYear,
	    endPrdDe = endYear,
	    )

	df = pd.DataFrame(df)
	df.to_csv('data/raw_data/IAI.csv', encoding='utf-8')
	return df

def collect_GT(keyword, targetCountry=''):
	"""-------------------------------------------------------------------
	구글 트렌드 데이터 수집

	Source: 
	PyTrend API = https://pypi.org/project/pytrends/
	Google Trends = https://trends.google.co.kr/trends?geo=US&hl=en-US
	"""
	# Initialize PyTrend
	pytrend = TrendReq(hl='en-US', tz=360)

	geo = coco.convert(
		names=targetCountry, src='UNCode', to='iso2'
	) if targetCountry else ''

	# Build payload
	pytrend.build_payload(kw_list=[keyword], timeframe='all', geo=geo, gprop='')

	# Get interest over time
	data = pytrend.interest_over_time()

	# Save to CSV
	data.to_csv(f'data/raw_data/GT_{keyword}_{geo}.csv', encoding='utf-8')
	return data

# Manually Downloaded Data
"""-------------------------------------------------------------------
ㅁ GDP 
World Bank Group = https://data.worldbank.org/indicator/NY.GDP.MKTP.CD (현재 이거 쓰는중)
KOSIS = https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3DR_SUB_UTITLE_L%26obj_var_id%3D%26seqNo%3D%26tblId%3DDT_2KAA903%26vw_cd%3DMT_RTITLE%26orgId%3D101%26path%3D%252FstatisticsList%252FstatisticsListIndex.do%26conn_path%3DMT_RTITLE%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26

ㅁ WTI
Investing.com = https://www.investing.com/commodities/crude-oil

ㅁ MOP
Investing.com = https://www.investing.com/currencies/usd-mop-historical-data

ㅁ EUP_IUP_TOT_ITOT
Trade Statistics Service = https://www.bandtrass.or.kr/analysis/unit.do?command=UNI001View&viewCode=UNI00102

ㅁ US Export Index Price
Federal Reserve Bank = https://fred.stlouisfed.org/categories/32225

Google search trend
Google Trends = https://trends.google.co.kr/trends?geo=US&hl=en-US
PyTrend API call (https://pypi.org/project/pytrends/)
"""

def collect_data(hsCodes, reporterCountry=410, partnerCountry=None, startYear=2014, endYear=2023):
	# 글로벌 데이터-----------------------------------------------------------------
	collect_TE()

	collect_CPI(startYear, endYear)

	collect_PPI(startYear, endYear)

	collect_IR(startYear, endYear)

	collect_ER(startYear, endYear)

	collect_EAI(startYear, endYear)

	collect_IAI(startYear, endYear)


	# 특정 데이터-----------------------------------------------------------------
	collect_TEV(startYear, endYear, hsCodes)

	# collect_GT('fruit', targetCountry=840)

if __name__ == "__main__":
	hsCodes = "10"  # Example HS Code for fruits
	reporterCountry = 410  # South Korea
	partnerCountry = None  # All partners
	startYear = 2014
	endYear = 2025

	# 글로벌 데이터-----------------------------------------------------------------
	# collect_TE()

	# collect_CPI(startYear, endYear)

	# collect_PPI(startYear, endYear)

	# collect_IR(startYear, endYear)

	# collect_ER(startYear, endYear)

	# collect_EAI(startYear, endYear)

	# collect_IAI(startYear, endYear)


	# 특정 데이터-----------------------------------------------------------------
	collect_TEV(startYear, endYear, hsCodes)

	collect_GT('fruit', targetCountry=840)

