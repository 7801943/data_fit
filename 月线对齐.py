import pandas as pd
import numpy as np
import tqdm

def merge_dataframes_by_date(start, end, dataframe1, time_column, data_column):
    # 创建新的base DataFrame
    new_base = pd.DataFrame({'日期': pd.date_range(start=start, end=end, freq='M')})

    # 将日期列转换为日期时间类型
    new_base['日期'] = pd.to_datetime(new_base['日期'])
    dataframe1[time_column] = pd.to_datetime(dataframe1[time_column])

    # 根据日期列对数据进行排序
    new_base.sort_values('日期', inplace=True)
    dataframe1.sort_values(time_column, inplace=True)

    # 获取dataframe1中按月份筛选后的日期列表
    dataframe1_months = dataframe1[time_column].dt.to_period('M').unique()

    # 合并数据
    merged_data = []
    for _, row in tqdm.tqdm(new_base.iterrows()):
        base_date = row['日期']
        base_month = base_date.to_period('M')

        if base_month in dataframe1_months:
            #选择出月份相同的日期
            matching_dates = dataframe1[time_column].loc[dataframe1[time_column].dt.to_period('M') == base_month]
            #选出差异最小的日期
            closest_date = matching_dates.loc[(matching_dates - base_date).abs().idxmin()]
            merged_data.append(dataframe1[data_column].loc[dataframe1[time_column] == closest_date].values[0])
        else:
            merged_data.append(np.nan)

    # 将合并后的数据赋值给新的base DataFrame的第2列
    new_base[data_column] = merged_data

    return new_base
    # 将新的base DataFrame与原始base DataFrame按日期合并
    #merged_base = pd.merge(base, new_base, on='日期', how='left')

    #return merged_base


# test = pd.DataFrame({'日期': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-4-25', '2023-4-13', '2023-4-26'],
#                            '数据': [115, 220, 310, 425, 413, 426]})

# start_date = dataframe1['日期'].min()
# end_date = pd.to_datetime('today')

df1 = pd.read_excel('上证50历史PE(000016.SH).xlsx')
df1 = df1[0:-6]


merged_dataframe = merge_dataframes_by_date('2000-1-1', 'today', df1, data_column='市盈率(TTM)', time_column='交易日期')
merged_dataframe.to_excel("上证50历史PE(000016.SH)-月.xlsx")

# df2 = pd.read_excel('中债国债1年到期利率走势分析.xlsx')
# df2 = df2[5:-4]
#
# merged_dataframe = merge_dataframes_by_date('2000-1-1', 'today', df2, data_column='Unnamed: 1', time_column='利率走势数据')
# merged_dataframe.to_excel("中债国债1年到期利率走势分析-月.xlsx")
# print(merged_dataframe)
