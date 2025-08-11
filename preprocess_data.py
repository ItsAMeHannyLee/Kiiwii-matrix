import sqlite3
import pandas as pd
import numpy as np
import os
import country_converter as coco
import json

# GDP : 연별로 통일

def process_TE(periods):
    # Read te_data.csv file
    data = pd.read_csv("data/raw_data/te_data.csv")

    # Find unique categories in 'Category' column
    categories = data['Category'].unique()

    # Create empty DataFrame to store results
    df = pd.DataFrame()

    # Insert periods as new columns
    # if end period is this year, then set to current month-1
    if periods[1] == pd.Timestamp.now().year:
        end_month = pd.Timestamp.now().month - 1
        if end_month < 1:
            end_month = 12
            periods[1] -= 1
    else:
        end_month = 12
    df['DateTime'] = pd.date_range(start=f"{periods[0]}-01-01", end=f"{periods[1]}-{end_month}-01", freq='MS')
    
    for category in categories:
        df_category = data[data['Category'] == category]
        countries = df_category['Country'].unique()

        for country in countries:
            df_temp = df_category[df_category['Country'] == country]
            df_temp = df_temp[['DateTime', 'Value', 'Frequency']].copy()
            df_temp.rename(columns={'Value': f"{category}_{country}"}, inplace=True)

            # Ensure 'DateTime' is datetime type in both DataFrames
            df_temp['DateTime'] = pd.to_datetime(df_temp['DateTime'])

            if df_temp['Frequency'].values[0] == 'Daily':
                # Pick a data point closest to the first day of each month and remove the rest of the month's data
                df_temp['Month'] = df_temp['DateTime'].dt.to_period('M').dt.to_timestamp()
                df_temp['abs_diff'] = (df_temp['DateTime'] - df_temp['Month']).abs()
                df_temp = df_temp.loc[df_temp.groupby('Month')['abs_diff'].idxmin()]
                df_temp['DateTime'] = df_temp['Month']
                df_temp = df_temp.drop(columns=['Month', 'abs_diff'])
            elif df_temp['Frequency'].values[0] == 'Monthly':
                # Change the day of the month to the first day
                df_temp['DateTime'] = df_temp['DateTime'].dt.to_period('M').dt.to_timestamp()
            elif df_temp['Frequency'].values[0] == 'Quarterly':
                # Change the day of the month to the first day of the quarter
                df_temp['DateTime'] = df_temp['DateTime'].dt.to_period('Q').dt.to_timestamp()
            elif df_temp['Frequency'].values[0] == 'Yearly':
                # Change the day of the month to the first day of the year
                df_temp['DateTime'] = df_temp['DateTime'].dt.to_period('Y').dt.to_timestamp()
                
            # Drop 'Frequency' column before merging to avoid duplicate columns
            df_temp = df_temp.drop(columns=['Frequency'])
            df = pd.merge(df, df_temp, on='DateTime', how='left')
    
    # Interpolate missing values
    df = df.interpolate(method='linear', limit_direction='both')

    # save the DataFrame to a CSV file
    output_dir = "data/refined_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "TE_refined.csv")
    df.to_csv(output_path, index=False)

    return df

def process_TEV(hs_code="08"):
    # conn = sqlite3.connect("data/raw_data/trade_data.db")/
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "data/raw_data", "TEV.db"))

    query = """
    SELECT period, partnerCode, cmdCode, SUM(primaryValue) as total_value
    from trade_data
    WHERE cmdCode = ? AND flowCode = 'X'
    GROUP BY period, partnerCode
    """
    
    df = pd.read_sql_query(query, conn, params=(hs_code,))

    df['date'] = pd.to_datetime(df['period'], format='%Y%m')
    df['year_month'] = df['date'].dt.strftime('%Y-%m')

    # Pivot the data: rows as year_month, columns as partnerCode, values as total_value
    df_pivot = df.pivot_table(
        index='year_month',
        columns='partnerCode',
        values='total_value',
        aggfunc='sum',
        fill_value=0
    )

    # Reset index to have year_month as a column if needed
    df_pivot = df_pivot.reset_index()
    
    # choose only the top 10 partners by total value
    top_partners = df_pivot.iloc[:, 1:].sum().nlargest(10).index
    df_pivot = df_pivot[['year_month'] + list(top_partners)]
    
    # Remove index
    df_pivot['year_month'] = pd.to_datetime(df_pivot['year_month'], format='%Y-%m')
    df_pivot.set_index('year_month', inplace=True)
    
    # Convert M49 country codes to ISO3
    df_pivot.columns = coco.convert(names=df_pivot.columns, src="UNcode", to="ISO3")
    df_pivot.columns = [f"TEV_{col}" for col in df_pivot.columns]
    
    df_pivot.to_csv('data/refined_data/TEV_refined.csv')
    # print(f"Preprocessed data for HS code {hs_code} saved to data/processed_features.csv and data/trade_data.csv")
    conn.close()

    return df_pivot

def process_GDP(targetCountries, periods):
    """
    Process raw 'World GDP' csv data file by extracting targetCountries over wanted years

    Parameters:
    targetCountries [str, str, ...]: List of target country in ISO3
    periods [startYear, endYear]: List of start/end year to search

    Return:
    'GDP.csv' file
    """
    # CSV 파일 로드
    df = pd.read_csv("data/raw_data/GDP.csv", skiprows=4)

    # 데이터 추출 나라
    countries = targetCountries

    # 필요한 연도 (2010~2024)
    years = [str(year) for year in range(periods[0], periods[1]+1)]
    columns = ['Country Name', 'Country Code'] + years

    # 데이터 필터링
    df_filtered = df[df['Country Code'].isin(countries)][columns]

    # 결측값 처리 (선형 보간)
    df_filtered[years] = df_filtered[years].interpolate(method='linear', axis=1, limit_direction='forward')

    # 피벗 테이블 생성 (국가를 컬럼으로)
    df_pivot = df_filtered.melt(id_vars=['Country Name', 'Country Code'], value_vars=years, var_name='Year', value_name='GDP')
    df_pivot = df_pivot.pivot_table(index='Year', columns='Country Code', values='GDP').reset_index()
    df_pivot['Year'] = pd.to_datetime(df_pivot['Year'], format='%Y')
    df_pivot = df_pivot.set_index('Year')
    df_pivot.columns = [f"GDP_{col}" for col in df_pivot.columns]

    # Create monthly index (start from first year, end at last year's December)
    start_date = df_pivot.index[0]
    end_date = f"{periods[1]}-12-01"
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex to monthly frequency
    df_monthly = df_pivot.reindex(monthly_index)

    # Interpolate linearly
    df_monthly = df_monthly.interpolate(method='linear')

    # Reset index and add year_month
    df_monthly.reset_index(inplace=True)
    df_monthly.rename(columns={'index': 'Year'}, inplace=True)

    # Export data to csv
    df_monthly.to_csv("data/refined_data/GDP_refined.csv", index=None) 
    
    return df_monthly

def process_CPI(targetCountries, periods):
    """
    Process CPI_by_country.csv to pivot by target countries and years.

    Parameters:
    targetCountries (list): List of ISO alpha-3 country codes (e.g., ['KOR', 'CHN']).
    periods (list): Start and end years (e.g., [2010, 2023]).

    Returns:
    pandas.DataFrame: Pivoted DataFrame with Year and CPI by country.
    """
    # CSV 파일 로드
    df = pd.read_csv("data/raw_data/CPI.csv")

    # KOSIS -> ISO3 JSON 맵 로드
    with open("data/KOSIS_국가_분류코드.json", "r") as f:
        json_data = json.load(f)

    # JSON에서 KOSIS 코드와 ISO3 코드 매핑 생성
    kosis_to_iso3 = {item["KOSIS"]: item["ISO3"] for item in json_data["keys"]}

    # 분류값ID1을 ISO3 코드로 변환
    df['iso_code'] = df['분류값ID1'].astype(str).map(kosis_to_iso3)

    # 매핑되지 않은 국가 확인
    if df['iso_code'].isna().any():
        print("Warning: Some KOSIS codes could not be mapped to ISO3 codes:")
        print(df[df['iso_code'].isna()]['분류값ID1'].unique())
    
    # targetCountries와 periods로 필터링
    df = df[df['iso_code'].isin(targetCountries) & 
            df['수록시점'].between(periods[0], periods[1])]

    # 필요한 컬럼 선택
    df = df[['iso_code', '수록시점', '수치값']]

    # 피벗 테이블 생성
    df_pivot = pd.pivot_table(
        df,
        index='수록시점',
        columns='iso_code',
        values='수치값',
        aggfunc='first'  # 동일 연도/국가의 첫 번째 값
    )

    # 실제 데이터에 존재하는 targetCountries만 선택
    available_countries = [country for country in targetCountries if country in df_pivot.columns]
    if len(available_countries) < len(targetCountries):
        missing_countries = set(targetCountries) - set(available_countries)
        print(f"Warning: The following countries are not in the data and will be ignored: {missing_countries}")
    
    # 컬럼 순서 지정 (존재하는 국가만)
    if available_countries:  # 데이터가 존재하는 경우에만 컬럼 지정
        df_pivot = df_pivot[available_countries]
    else:
        print("No data available for the specified countries.")
        return pd.DataFrame()  # 빈 DataFrame 반환

    # 인덱스를 Year 컬럼으로
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'수록시점': 'Year'}, inplace=True)
    df_pivot['Year'] = pd.to_datetime(df_pivot['Year'], format='%Y')
    df_pivot = df_pivot.set_index('Year')
    df_pivot.columns = [f"CPI_{col}" for col in df_pivot.columns]

    # Create monthly index (start from first year, end at last year's December)
    start_date = df_pivot.index[0]
    end_date = f"{periods[1]}-12-01"
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex to monthly frequency
    df_monthly = df_pivot.reindex(monthly_index)

    # Interpolate linearly
    df_monthly = df_monthly.interpolate(method='linear')

    # Reset index and add year_month
    df_monthly.reset_index(inplace=True)
    df_monthly.rename(columns={'index': 'Year'}, inplace=True)

    # 결과 저장
    df_monthly.to_csv('data/refined_data/CPI_refined.csv', index=False)

    return df_pivot

def process_PPI(targetCountries, periods):
    """
    Process PPI_by_country.csv to pivot by target countries and years.

    Parameters:
    targetCountries (list): List of ISO alpha-3 country codes (e.g., ['KOR', 'CHN']).
    periods (list): Start and end years (e.g., [2010, 2023]).

    Returns:
    pandas.DataFrame: Pivoted DataFrame with Year and PPI by country.
    """
    # CSV 파일 로드
    df = pd.read_csv("data/raw_data/PPI.csv")

    # KOSIS -> ISO3 JSON 맵 로드
    with open("data/KOSIS_국가_분류코드.json", "r") as f:
        json_data = json.load(f)

    # JSON에서 KOSIS 코드와 ISO3 코드 매핑 생성
    kosis_to_iso3 = {item["KOSIS"]: item["ISO3"] for item in json_data["keys"]}

    # 분류값ID1을 ISO3 코드로 변환
    df['iso_code'] = df['분류값ID1'].astype(str).map(kosis_to_iso3)

    # 매핑되지 않은 국가 확인
    if df['iso_code'].isna().any():
        print("Warning: Some KOSIS codes could not be mapped to ISO3 codes:")
        print(df[df['iso_code'].isna()]['분류값ID1'].unique())
    
    # targetCountries와 periods로 필터링
    df = df[df['iso_code'].isin(targetCountries) & 
            df['수록시점'].between(periods[0], periods[1])]

    # 필요한 컬럼 선택
    df = df[['iso_code', '수록시점', '수치값']]

    # 피벗 테이블 생성
    df_pivot = pd.pivot_table(
        df,
        index='수록시점',
        columns='iso_code',
        values='수치값',
        aggfunc='first'  # 동일 연도/국가의 첫 번째 값
    )

    # 실제 데이터에 존재하는 targetCountries만 선택
    available_countries = [country for country in targetCountries if country in df_pivot.columns]
    if len(available_countries) < len(targetCountries):
        missing_countries = set(targetCountries) - set(available_countries)
        print(f"Warning: The following countries are not in the data and will be ignored: {missing_countries}")
    
    # 컬럼 순서 지정 (존재하는 국가만)
    if available_countries:  # 데이터가 존재하는 경우에만 컬럼 지정
        df_pivot = df_pivot[available_countries]
    else:
        print("No data available for the specified countries.")
        return pd.DataFrame()  # 빈 DataFrame 반환

    # 인덱스를 Year 컬럼으로
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'수록시점': 'Year'}, inplace=True)
    df_pivot['Year'] = pd.to_datetime(df_pivot['Year'], format='%Y')
    df_pivot = df_pivot.set_index('Year')
    df_pivot.columns = [f"PPI_{col}" for col in df_pivot.columns]

    # Create monthly index (start from first year, end at last year's December)
    start_date = df_pivot.index[0]
    end_date = f"{periods[1]}-12-01"
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex to monthly frequency
    df_monthly = df_pivot.reindex(monthly_index)

    # Interpolate linearly
    df_monthly = df_monthly.interpolate(method='linear')

    # Reset index and add year_month
    df_monthly.reset_index(inplace=True)
    df_monthly.rename(columns={'index': 'Year'}, inplace=True)

    # 결과 저장
    df_monthly.to_csv('data/refined_data/PPI_refined.csv', index=False)

    return df_pivot

def process_IR(targetCountries, periods):
    """
    Process IR_by_country.csv to pivot by target countries and years.

    Parameters:
    targetCountries (list): List of ISO alpha-3 country codes (e.g., ['KOR', 'CHN']).
    periods (list): Start and end years (e.g., [2010, 2023]).

    Returns:
    pandas.DataFrame: Pivoted DataFrame with Year and IR by country.
    """
    # CSV 파일 로드
    df = pd.read_csv("data/raw_data/IR.csv")

    # KOSIS -> ISO3 JSON 맵 로드
    with open("data/KOSIS_국가_분류코드.json", "r") as f:
        json_data = json.load(f)

    # JSON에서 KOSIS 코드와 ISO3 코드 매핑 생성
    kosis_to_iso3 = {item["KOSIS"]: item["ISO3"] for item in json_data["keys"]}

    # 분류값ID1을 ISO3 코드로 변환
    df['iso_code'] = df['분류값ID1'].astype(str).map(kosis_to_iso3)

    # 매핑되지 않은 국가 확인
    if df['iso_code'].isna().any():
        print("Warning: Some KOSIS codes could not be mapped to ISO3 codes:")
        print(df[df['iso_code'].isna()]['분류값ID1'].unique())
    
    # targetCountries와 periods로 필터링
    df = df[df['iso_code'].isin(targetCountries) & 
            df['수록시점'].between(periods[0], periods[1])]

    # 필요한 컬럼 선택
    df = df[['iso_code', '수록시점', '수치값']]

    # 피벗 테이블 생성
    df_pivot = pd.pivot_table(
        df,
        index='수록시점',
        columns='iso_code',
        values='수치값',
        aggfunc='first'  # 동일 연도/국가의 첫 번째 값
    )

    # 실제 데이터에 존재하는 targetCountries만 선택
    available_countries = [country for country in targetCountries if country in df_pivot.columns]
    if len(available_countries) < len(targetCountries):
        missing_countries = set(targetCountries) - set(available_countries)
        print(f"Warning: The following countries are not in the data and will be ignored: {missing_countries}")
    
    # 컬럼 순서 지정 (존재하는 국가만)
    if available_countries:  # 데이터가 존재하는 경우에만 컬럼 지정
        df_pivot = df_pivot[available_countries]
    else:
        print("No data available for the specified countries.")
        return pd.DataFrame()  # 빈 DataFrame 반환

    # 인덱스를 Year 컬럼으로
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'수록시점': 'Year'}, inplace=True)
    df_pivot['Year'] = pd.to_datetime(df_pivot['Year'], format='%Y')
    df_pivot = df_pivot.set_index('Year')
    df_pivot.columns = [f"IR_{col}" for col in df_pivot.columns]

    # Create monthly index (start from first year, end at last year's December)
    start_date = df_pivot.index[0]
    end_date = f"{periods[1]}-12-01"
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex to monthly frequency
    df_monthly = df_pivot.reindex(monthly_index)

    # Interpolate linearly
    df_monthly = df_monthly.interpolate(method='linear')

    # Reset index and add year_month
    df_monthly.reset_index(inplace=True)
    df_monthly.rename(columns={'index': 'Year'}, inplace=True)

    # 결과 저장
    df_monthly.to_csv('data/refined_data/IR_refined.csv', index=False)

    return df_pivot

def process_ER(targetCountries, periods):
    """
    Process ExchangeRate_by_country.csv to pivot by target countries and years.

    Parameters:
    targetCountries (list): List of ISO alpha-3 country codes (e.g., ['KOR', 'CHN']).
    periods (list): Start and end years (e.g., [2010, 2023]).

    Returns:
    pandas.DataFrame: Pivoted DataFrame with Year and ExchangeRate by country.
    """
    # CSV 파일 로드
    df = pd.read_csv("data/raw_data/ER.csv")

    # KOSIS -> ISO3 JSON 맵 로드
    with open("data/KOSIS_국가_분류코드.json", "r") as f:
        json_data = json.load(f)

    # JSON에서 KOSIS 코드와 ISO3 코드 매핑 생성
    kosis_to_iso3 = {item["KOSIS"]: item["ISO3"] for item in json_data["keys"]}

    # 분류값ID1을 ISO3 코드로 변환
    df['iso_code'] = df['분류값ID1'].astype(str).map(kosis_to_iso3)

    # 매핑되지 않은 국가 확인
    if df['iso_code'].isna().any():
        print("Warning: Some KOSIS codes could not be mapped to ISO3 codes:")
        print(df[df['iso_code'].isna()]['분류값ID1'].unique())
    
    # targetCountries와 periods로 필터링
    df = df[df['iso_code'].isin(targetCountries) & 
            df['수록시점'].between(periods[0], periods[1])]

    # 필요한 컬럼 선택
    df = df[['iso_code', '수록시점', '수치값']]

    # 피벗 테이블 생성
    df_pivot = pd.pivot_table(
        df,
        index='수록시점',
        columns='iso_code',
        values='수치값',
        aggfunc='first'  # 동일 연도/국가의 첫 번째 값
    )

    # 실제 데이터에 존재하는 targetCountries만 선택
    available_countries = [country for country in targetCountries if country in df_pivot.columns]
    if len(available_countries) < len(targetCountries):
        missing_countries = set(targetCountries) - set(available_countries)
        print(f"Warning: The following countries are not in the data and will be ignored: {missing_countries}")
    
    # 미국 USD가 포함되어있다면 제외
    if 'USD' in available_countries:
        available_countries.remove('USD')
        print("USD is excluded from the analysis as it is the base currency.")

    # 컬럼 순서 지정 (존재하는 국가만)
    if available_countries:  # 데이터가 존재하는 경우에만 컬럼 지정
        df_pivot = df_pivot[available_countries]
    else:
        print("No data available for the specified countries.")
        return pd.DataFrame()  # 빈 DataFrame 반환

    # 인덱스를 Year 컬럼으로
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'수록시점': 'Year'}, inplace=True)
    df_pivot['Year'] = pd.to_datetime(df_pivot['Year'], format='%Y')
    df_pivot = df_pivot.set_index('Year')
    df_pivot.columns = [f"ER_{col}" for col in df_pivot.columns]

    # Create monthly index (start from first year, end at last year's December)
    start_date = df_pivot.index[0]
    end_date = f"{periods[1]}-12-01"
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex to monthly frequency
    df_monthly = df_pivot.reindex(monthly_index)

    # Interpolate linearly
    df_monthly = df_monthly.interpolate(method='linear')

    # Reset index and add year_month
    df_monthly.reset_index(inplace=True)
    df_monthly.rename(columns={'index': 'Year'}, inplace=True)

    # 결과 저장
    df_monthly.to_csv('data/refined_data/ER_refined.csv', index=False)

    return df_pivot

def process_EAI(targetCountries, periods):
    """
    Process EAI.csv to pivot by target countries and years.

    Parameters:
    targetCountries (list): List of ISO alpha-3 country codes (e.g., ['KOR', 'CHN']).
    periods (list): Start and end years (e.g., [2010, 2023]).

    Returns:
    pandas.DataFrame: Pivoted DataFrame with Year and Export Amount Index by country.
    """
    # CSV 파일 로드
    df = pd.read_csv("data/raw_data/EAI.csv")

    # KOSIS -> ISO3 JSON 맵 로드
    with open("data/KOSIS_국가_분류코드.json", "r") as f:
        json_data = json.load(f)

    # JSON에서 KOSIS 코드와 ISO3 코드 매핑 생성
    kosis_to_iso3 = {item["KOSIS"]: item["ISO3"] for item in json_data["keys"]}

    # 분류값ID1을 ISO3 코드로 변환
    df['iso_code'] = df['분류값ID1'].astype(str).map(kosis_to_iso3)

    # 매핑되지 않은 국가 확인
    if df['iso_code'].isna().any():
        print("Warning: Some KOSIS codes could not be mapped to ISO3 codes:")
        print(df[df['iso_code'].isna()]['분류값ID1'].unique())
    
    # targetCountries와 periods로 필터링
    df = df[df['iso_code'].isin(targetCountries) & 
            df['수록시점'].between(periods[0], periods[1])]

    # 필요한 컬럼 선택
    df = df[['iso_code', '수록시점', '수치값']]

    # 피벗 테이블 생성
    df_pivot = pd.pivot_table(
        df,
        index='수록시점',
        columns='iso_code',
        values='수치값',
        aggfunc='first'  # 동일 연도/국가의 첫 번째 값
    )

    # 실제 데이터에 존재하는 targetCountries만 선택
    available_countries = [country for country in targetCountries if country in df_pivot.columns]
    if len(available_countries) < len(targetCountries):
        missing_countries = set(targetCountries) - set(available_countries)
        print(f"Warning: The following countries are not in the data and will be ignored: {missing_countries}")
    
    # 컬럼 순서 지정 (존재하는 국가만)
    if available_countries:  # 데이터가 존재하는 경우에만 컬럼 지정
        df_pivot = df_pivot[available_countries]
    else:
        print("No data available for the specified countries.")
        return pd.DataFrame()  # 빈 DataFrame 반환

    # 인덱스를 Year 컬럼으로
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'수록시점': 'Year'}, inplace=True)
    df_pivot['Year'] = pd.to_datetime(df_pivot['Year'], format='%Y')
    df_pivot = df_pivot.set_index('Year')
    df_pivot.columns = [f"EAI_{col}" for col in df_pivot.columns]

    # Create monthly index (start from first year, end at last year's December)
    start_date = df_pivot.index[0]
    end_date = f"{periods[1]}-12-01"
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex to monthly frequency
    df_monthly = df_pivot.reindex(monthly_index)

    # Interpolate linearly
    df_monthly = df_monthly.interpolate(method='linear')

    # Reset index and add year_month
    df_monthly.reset_index(inplace=True)
    df_monthly.rename(columns={'index': 'Year'}, inplace=True)

    # 결과 저장
    df_monthly.to_csv('data/refined_data/EAI_refined.csv', index=False)

    return df_pivot

def process_IAI(targetCountries, periods):
    """
    Process IAI_by_country.csv to pivot by target countries and years.

    Parameters:
    targetCountries (list): List of ISO alpha-3 country codes (e.g., ['KOR', 'CHN']).
    periods (list): Start and end years (e.g., [2010, 2023]).

    Returns:
    pandas.DataFrame: Pivoted DataFrame with Year and Import Amount Index by country.
    """
    # CSV 파일 로드
    df = pd.read_csv("data/raw_data/IAI.csv")

    # KOSIS -> ISO3 JSON 맵 로드
    with open("data/KOSIS_국가_분류코드.json", "r") as f:
        json_data = json.load(f)

    # JSON에서 KOSIS 코드와 ISO3 코드 매핑 생성
    kosis_to_iso3 = {item["KOSIS"]: item["ISO3"] for item in json_data["keys"]}

    # 분류값ID1을 ISO3 코드로 변환
    df['iso_code'] = df['분류값ID1'].astype(str).map(kosis_to_iso3)

    # 매핑되지 않은 국가 확인
    if df['iso_code'].isna().any():
        print("Warning: Some KOSIS codes could not be mapped to ISO3 codes:")
        print(df[df['iso_code'].isna()]['분류값ID1'].unique())
    
    # targetCountries와 periods로 필터링
    df = df[df['iso_code'].isin(targetCountries) & 
            df['수록시점'].between(periods[0], periods[1])]

    # 필요한 컬럼 선택
    df = df[['iso_code', '수록시점', '수치값']]

    # 피벗 테이블 생성
    df_pivot = pd.pivot_table(
        df,
        index='수록시점',
        columns='iso_code',
        values='수치값',
        aggfunc='first'  # 동일 연도/국가의 첫 번째 값
    )

    # 실제 데이터에 존재하는 targetCountries만 선택
    available_countries = [country for country in targetCountries if country in df_pivot.columns]
    if len(available_countries) < len(targetCountries):
        missing_countries = set(targetCountries) - set(available_countries)
        print(f"Warning: The following countries are not in the data and will be ignored: {missing_countries}")
    
    # 컬럼 순서 지정 (존재하는 국가만)
    if available_countries:  # 데이터가 존재하는 경우에만 컬럼 지정
        df_pivot = df_pivot[available_countries]
    else:
        print("No data available for the specified countries.")
        return pd.DataFrame()  # 빈 DataFrame 반환

    # 인덱스를 Year 컬럼으로
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'수록시점': 'Year'}, inplace=True)
    df_pivot['Year'] = pd.to_datetime(df_pivot['Year'], format='%Y')
    df_pivot = df_pivot.set_index('Year')
    df_pivot.columns = [f"IAI_{col}" for col in df_pivot.columns]

    # Create monthly index (start from first year, end at last year's December)
    start_date = df_pivot.index[0]
    end_date = f"{periods[1]}-12-01"
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex to monthly frequency
    df_monthly = df_pivot.reindex(monthly_index)

    # Interpolate linearly
    df_monthly = df_monthly.interpolate(method='linear')

    # Reset index and add year_month
    df_monthly.reset_index(inplace=True)
    df_monthly.rename(columns={'index': 'Year'}, inplace=True)

    # 결과 저장
    df_monthly.to_csv('data/refined_data/IAI_refined.csv', index=False)

    return df_pivot

def process_WTI(periods):
    """
    Extract WTI Crude Oil Futures price data for the specified year range.

    Parameters:
    periods (list): Start and end years (e.g., [2000, 2023]).

    Returns:
    pandas.DataFrame: DataFrame with Date and Price columns for the specified years.
    """
    # 파일 경로
    file_path = "data/raw_data/WTI.csv"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()

    try:
        # 엑셀 파일 로드
        df = pd.read_csv(file_path)

        # Date 컬럼을 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

        # 연도 추출
        df['Year'] = df['Date'].dt.year

        # periods로 필터링
        start_year, end_year = periods
        df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

        # 필요한 컬럼 선택
        df_filtered = df_filtered[['Date', 'Price']]

        # 데이터가 없는 경우 경고
        if df_filtered.empty:
            print(f"Warning: No data found for the year range {start_year}–{end_year}.")
            return pd.DataFrame()
        
        # Date 기준 오름차순 정렬
        df_filtered = df_filtered.sort_values('Date', ascending=True)
        df_filtered.columns = [f"WTI_{col}" for col in df_filtered.columns]

        # 결과 저장
        output_dir = "data/refined_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/WTI_refined.csv"
        df_filtered.to_csv(output_path, index=False)

        # print(f"WTI price data for {start_year}–{end_year} saved to {output_path}")

        return df_filtered

    except Exception as e:
        print(f"Error processing WTI data: {str(e)}")
        return pd.DataFrame()

def process_MOP(periods):
    """
    Extract MOP Futures price data for the specified year range.

    Parameters:
    periods (list): Start and end years (e.g., [2000, 2023]).

    Returns:
    pandas.DataFrame: DataFrame with Date and Price columns for the specified years.
    """
    # 파일 경로
    file_path = "data/raw_data/MOP.csv"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()

    try:
        # 엑셀 파일 로드
        df = pd.read_csv(file_path)

        # Date 컬럼을 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

        # 연도 추출
        df['Year'] = df['Date'].dt.year

        # periods로 필터링
        start_year, end_year = periods
        df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

        # 필요한 컬럼 선택
        df_filtered = df_filtered[['Date', 'Price']]

        # 데이터가 없는 경우 경고
        if df_filtered.empty:
            print(f"Warning: No data found for the year range {start_year}–{end_year}.")
            return pd.DataFrame()
        
        # Date 기준 오름차순 정렬
        df_filtered = df_filtered.sort_values('Date', ascending=True)
        df_filtered.columns = [f"MOP_{col}" for col in df_filtered.columns]

        # 결과 저장
        output_dir = "data/refined_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/MOP_refined.csv"
        df_filtered.to_csv(output_path, index=False)

        # print(f"WTI price data for {start_year}–{end_year} saved to {output_path}")

        return df_filtered

    except Exception as e:
        print(f"Error processing WTI data: {str(e)}")
        return pd.DataFrame()

def process_EUP_IUP_TOT_ITOT(periods):
    """
    Extract trade indices (Export/Import Price, Net/Income Terms of Trade) for the specified year range,
    sorted by Date in ascending order.

    Parameters:
    periods (list): Start and end years (e.g., [2000, 2023]).

    Returns:
    pandas.DataFrame: DataFrame with Date and trade indices, sorted by Date.
    """
    # 파일 경로
    file_path = "data/raw_data/EUP_IUP_TOT_ITOT.xlsx"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()

    try:
        # 엑셀 파일 로드
        df = pd.read_excel(file_path, skiprows=3)

        # 데이터 전처리: 품목명을 인덱스로 설정
        df.set_index('품목명', inplace=True)

        # 고정 컬럼(단위, 항목명1, 항목명2) 제외, 날짜 컬럼만 선택
        date_columns = [col for col in df.columns if col.startswith('20')]
        df = df.loc[:, date_columns]

        # 데이터를 전치하여 날짜를 행으로 변환
        df = df.transpose()

        # 인덱스를 Date 컬럼으로 변환 (YYYYMM -> YYYY-MM-01)
        df.index = pd.to_datetime(df.index, format='%Y%m').strftime('%Y-%m-01')
        df.index.name = 'Date'

        # 컬럼명 영문화
        df.columns = [
            'Export_Price_Index',
            'Import_Price_Index',
            'Net_Terms_of_Trade_Index',
            'Income_Terms_of_Trade_Index'
        ]

        # DataFrame으로 변환
        df = df.reset_index()

        # Date를 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'])

        # 연도 추출
        df['Year'] = df['Date'].dt.year

        # periods로 필터링
        start_year, end_year = periods
        df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

        # 데이터가 없는 경우 경고
        if df_filtered.empty:
            print(f"Warning: No data found for the year range {start_year}–{end_year}.")
            return pd.DataFrame()

        # Date 기준 오름차순 정렬
        df_filtered = df_filtered.sort_values('Date', ascending=True)

        # Year 컬럼 제거
        df_filtered = df_filtered[['Date', 'Export_Price_Index', 'Import_Price_Index',
                                  'Net_Terms_of_Trade_Index', 'Income_Terms_of_Trade_Index']]
        
        # column 이름에 따라 이름 변경 (Export_Price_Index-> EUP, Import_Price_Index-> IUP, Net_Terms_of_Trade_Index-> TOT, Income_Terms_of_Trade_Index-> ITOT)
        df_filtered.columns = ['Date', 'EUP', 'IUP', 'TOT', 'ITOT']
        
        # 결과 저장
        output_dir = "data/refined_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/EUP_IUP_TOT_ITOT_refined.csv"
        df_filtered.to_csv(output_path, index=False)

        # print(f"Trade indices data for {start_year}–{end_year} saved to {output_path}")
        # print(df_filtered.head())

        return df_filtered

    except Exception as e:
        print(f"Error processing trade indices data: {str(e)}")
        return pd.DataFrame()

def process_GT(periods, keyword):
    """
    Process Google Trend data for Apple in the USA for the specified year range.

    Args:
        periods (list): Start and end years (e.g., [2010, 2023]).

    Returns:
        pandas.DataFrame: DataFrame with Month and Interest columns for the specified years.
    """
    # 파일 경로
    file_path = f"data/raw_data/GT_{keyword}_US.csv"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()
    
    try:
        # 엑셀 파일 로드
        df = pd.read_csv(file_path, skiprows=0)
        # 'Month' 컬럼을 datetime으로 변환
        df['Month'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        # 연도 추출
        df['Year'] = df['Month'].dt.year
        # periods로 필터링
        start_year, end_year = periods
        df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
        # 데이터가 없는 경우 경고
        if df_filtered.empty:
            print(f"Warning: No data found for the year range {start_year}–{end_year}.")
            return pd.DataFrame()
        # Date 기준 오름차순 정렬
        df_filtered = df_filtered.sort_values('Month', ascending=True)
        # 필요한 컬럼 선택
        df_filtered = df_filtered.iloc[:, 0:2]
        # Reset index
        df_filtered.reset_index(drop=True, inplace=True)
        # 컬럼 이름 변경
        df_filtered.columns = ['Month', f'GT_{keyword}_US']
        
        # 결과 저장
        output_dir = "data/refined_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/GT_refined.csv"
        df_filtered.to_csv(output_path, index=False)
        
        # print(f"Google Trend data for {start_year}–{end_year} saved to {output_path}")
        return df_filtered
    
    except Exception as e:
        print(f"Error processing Google Trend data: {str(e)}")
        return pd.DataFrame()
    

    """
    Extract Agricultural Import Price Index for the specified year range, sorted by Date in ascending order.

    Parameters:
    periods (list): Start and end years (e.g., [2000, 2023]).

    Returns:
    pandas.DataFrame: DataFrame with Date and Agri_Import_Price_Index, sorted by Date.
    """
    # 파일 경로
    file_path = "data/raw_data/Agri_Import_Price.xlsx"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()

    try:
        # 엑셀 파일 로드
        df = pd.read_excel(file_path, skiprows=3)

        # '농산물' 품목만 선택
        df = df[df['품목명'] == '농산물']

        # 고정 컬럼(단위, 항목명1, 항목명2) 제외, 날짜 컬럼만 선택
        date_columns = [col for col in df.columns if col.startswith('20')]
        df = df.loc[:, date_columns]

        # 데이터를 전치하여 날짜를 행으로 변환
        df = df.transpose()

        # 인덱스를 Date 컬럼으로 변환 (YYYYMM -> YYYY-MM-01)
        df.index = pd.to_datetime(df.index, format='%Y%m').strftime('%Y-%m-01')
        df.index.name = 'Date'

        # 컬럼명 설정
        df.columns = ['Agri_Import_Price_Index']

        # DataFrame으로 변환
        df = df.reset_index()

        # Date를 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'])

        # 연도 추출
        df['Year'] = df['Date'].dt.year

        # periods로 필터링
        start_year, end_year = periods
        df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

        # 데이터가 없는 경우 경고
        if df_filtered.empty:
            print(f"Warning: No data found for the year range {start_year}–{end_year}.")
            return pd.DataFrame()

        # 0 값 비율 확인
        zero_count = (df_filtered['Agri_Import_Price_Index'] == 0).sum()
        if zero_count > len(df_filtered) * 0.5:
            print(f"Warning: {zero_count}/{len(df_filtered)} entries are 0. Data may be incomplete for {start_year}–{end_year}.")

        # Date 기준 오름차순 정렬
        df_filtered = df_filtered.sort_values('Date', ascending=True)

        # Year 컬럼 제거
        df_filtered = df_filtered[['Date', 'Agri_Import_Price_Index']]

        # 결과 저장
        output_dir = "data/refined_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/agri_import_price.csv"
        df_filtered.to_csv(output_path, index=False)

        # print(f"Agricultural import price index data for {start_year}–{end_year} saved to {output_path}")
        # print(df_filtered.head())

        return df_filtered

    except Exception as e:
        print(f"Error processing agricultural import price data: {str(e)}")
        return pd.DataFrame()

# Combine all processed dataframes into a single CSV file according to the data's time.
def combine_dataframes(periods, target_periods):
    """
    Combine all processed dataframes into a single CSV file.

    Returns:
        None
    """
    # Load all processed dataframes
    files = [
        'GT_refined.csv',
        'PPI_refined.csv',
        'IR_refined.csv',
        'ER_refined.csv',
        'EAI_refined.csv',
        'IAI_refined.csv',
        'WTI_refined.csv',
        'MOP_refined.csv',
        'EUP_IUP_TOT_ITOT_refined.csv',
        'TEV_refined.csv',
        'GDP_refined.csv',
        'CPI_refined.csv',
        'TE_refined.csv',
    ]

    combined_df = pd.DataFrame()
    for file in files:
        file_path = f'data/refined_data/{file}'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df.iloc[:,1:]], axis=1)

    # Add t-1 to t-5 columns for each data column using pd.concat for better performance
    shifted_cols = {}
    for col in combined_df.columns:
        for i in range(1, 3):
            new_col_name = f"t-{i}_{col}"
            shifted_cols[new_col_name] = combined_df[col].shift(i)
    if shifted_cols:
        shifted_df = pd.DataFrame(shifted_cols)
        combined_df = pd.concat([combined_df, shifted_df], axis=1)

    # Extrapolate data to fill in missing months for t-1 to t-5
    t_minus_columns = [col for col in combined_df.columns if col.startswith('t-')]
    combined_df[t_minus_columns] = combined_df[t_minus_columns].interpolate(method='linear', axis=0, limit_direction='both')
    
    month_series = pd.Series(pd.date_range(start=f"{periods[0]}-01-01", end=f"{periods[1]}-12-31", freq='MS'), name='Month')
    combined_df = pd.concat([month_series, combined_df], axis=1)
    
    # Round all data to 2 decimal places
    combined_df = combined_df.round(2)

    # Filter the combined dataframe to only include the target periods
    start_date = pd.to_datetime(f"{target_periods[0]}-01-01")
    end_date = pd.to_datetime(f"{target_periods[1]}-12-31")
    target_df = combined_df[(combined_df['Month'] >= start_date) & (combined_df['Month'] <= end_date)]

    # convert values to float, forcing errors to NaN
    for col in target_df.columns[1:]:
        target_df.loc[:, col] = pd.to_numeric(target_df[col], errors='coerce')
    
    # Fill NaN values with 0
    target_df.fillna(0, inplace=True)

    # Save combined dataframe to CSV
    output_dir = "data/combined_data"
    os.makedirs(output_dir, exist_ok=True)
    combined_df.to_csv(f"{output_dir}/combined_data.csv", index=False)
    target_df.to_csv(f"{output_dir}/target_combined_data.csv", index=False)

if __name__ == "__main__":
    targetCountries = ['KOR', 'CHN', 'USA', 'HKG', 'JPN', 'VNM', 'IND', 'AUS', 'MEX']
    periods = [2016, 2023]

    # Trade Data
    process_TEV()

    # GDP
    # process_GDP(targetCountries, periods)

    # CPI: 소비자 물가지수
    # process_CPI(targetCountries, periods)

    # PPI: 생산자 물가지수
    # process_PPI(targetCountries, periods)

    # IR: 금리
    # process_IR(targetCountries, periods)

    # # Exchange Rate: 환율
    process_ER(targetCountries, periods)

    # Export Amount Index: 수출 물량지수
    process_EAI(targetCountries, periods)

    # Import Amount Index: 수입 물량지수
    process_IAI(targetCountries, periods)

    # WTI: 유가
    process_WTI(periods)

    # MOP: 유가
    process_MOP(periods)

    # Trade Indices: EPI/ IPI/ Net Terms of Trade Index / Income Terms of Trade Index
    process_EUP_IUP_TOT_ITOT(periods)
    
    # Google Trend Data
    process_GT(periods, 'coffee')

    # GDP, CPI, PPI, IR, EPI, IPI
    process_TE(periods)
    
    # Combine all processed dataframes into a single CSV file
    combine_dataframes(periods, [2014, 2023])


    