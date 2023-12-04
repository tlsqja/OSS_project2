from pandas import *

data = read_csv("2019_kbo_for_kaggle_v2.csv")


def requirements_1():
    # TODO: Print the top 10 players in H, avg, HR, and OBP for each year from 2015 to 2018.
    print("==================== Requirements #1 ====================\n")

    for year in range(2015, 2019):
        df_of_year = data[data['year'] == year]
        # 특정 시즌의 data 추출

        print(f"\n\nIn {year}")
        # 시즌 표시
        h_df = df_of_year.sort_index().sort_values(by='H', ascending=False).groupby('year').head(10)
        avg_df = df_of_year.sort_index().sort_values(by='avg', ascending=False).groupby('year').head(10)
        hr_df = df_of_year.sort_index().sort_values(by='HR', ascending=False).groupby('year').head(10)
        obp_df = df_of_year.sort_index().sort_values(by='OBP', ascending=False).groupby('year').head(10)
        # index 에 대하여 정렬 후 값에 대해 정렬, 우선 순위 10개를 가지는 dataframe 생성
        # 값이 같은 data 에 대하여 index 가 작은 값을 우선 순위에 두기 위함

        print(f"\nTop 10 players in hits\n{h_df[['batter_name', 'H']]}")
        print(f"\nTop 10 players in batting average\n{avg_df[['batter_name', 'avg']]}")
        print(f"\nTop 10 players in homerun\n{hr_df[['batter_name', 'HR']]}")
        print(f"\nTop 10 players in on-base percentage\n{obp_df[['batter_name', 'OBP']]}")
        # 결과에 대하여 dataframe 형태로 출력


def requirements_2():
    # TODO: Print the player with the highest war by position in 2018.
    print("\n\n==================== Requirements #2 ====================\n")

    df_of_2018 = data[data['year'] == 2018]
    # 2018년도 data 를 가지는 dataframe 생성
    positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    # position list 생성

    for position in positions:
        pc_df = df_of_2018[df_of_2018['cp'] == position]
        # 특정 position 의 data 를 가지는 dataframe 생성
        highest_war_player = pc_df.sort_index().sort_values(by='war', ascending=False).head(1)
        print(f"\nPlayer with the highest war in {position} in 2018\n{highest_war_player[['batter_name', 'war']]}\n")
        # batter_name 과 war 을 column 으로 가지는 dataframe 형태로 출력


def requirements_3():
    # TODO: Among R, H, HR, RBI, SB, war, avg, OBP, and SLG, which has the highest correlation with salary?
    # TODO: Implement code to calculate correlations and print the answer to the above question.
    print("\n\n==================== Requirements #3 ====================\n")

    columns = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
    # salary 와 상관 관계를 계산할 columns list 생성
    corr_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # correlation value 를 저장할 list 생성

    for i in range(9):
        corr_list[i] = data['salary'].corr(data[columns[i]])
        # salary 와 correlation 계산

    corr_df = DataFrame({'Data_name': columns, 'Correlation_with_salary': corr_list})
    print(f"\n{corr_df}\n")
    # 계산 결과 Data_name 과 Correlation_with_salary 를 column 으로 가지는 dataframe 형태로 출력

    max_corr = corr_list[0]
    idx = 0
    for i in range(1, 9):
        if max_corr < corr_list[i]:
            max_corr = corr_list[i]
            idx = i
    # correlation value 가 가장 큰 값 찾기

    print(f"\nThe highest correlation with salary is {columns[idx]}\n")
    # 결과값 출력


# 함수 참조
requirements_1()
requirements_2()
requirements_3()
