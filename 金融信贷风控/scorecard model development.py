import collections
import datetime
import numbers
import random
import sys
from importlib import reload

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

reload(sys)


# -*- coding: utf-8 -*-

################################
######## UDF: 自定义函数 ########
################################
### 对时间窗口，计算累计产比 ###
def TimeWindowSelection(df, daysCol, time_windows):
    '''
    :param df: the dataset containg variabel of days
    :param daysCol: the column of days
    :param time_windows: the list of time window
    :return:
    '''
    freq_tw = {}
    for tw in time_windows:
        freq = sum(df[daysCol].apply(lambda x: int(x <= tw)))
        freq_tw[tw] = freq
    return freq_tw


def DeivdedByZero(nominator, denominator):
    '''
    当分母为0时，返回0；否则返回正常值
    '''
    if denominator == 0:
        return 0
    else:
        return nominator * 1.0 / denominator


# 对某些统一的字段进行统一
def ChangeContent(x):
    y = x.upper()
    if y == '_MOBILEPHONE':
        y = '_PHONE'
    return y


def MissingCategorial(df, x):
    missing_vals = df[x].map(lambda x: int(x != x))
    return sum(missing_vals) * 1.0 / df.shape[0]


def MissingContinuous(df, x):
    missing_vals = df[x].map(lambda x: int(np.isnan(x)))
    return sum(missing_vals) * 1.0 / df.shape[0]


def MakeupRandom(x, sampledList):
    if x == x:
        return x
    else:
        randIndex = random.randint(0, len(sampledList) - 1)
        return sampledList[randIndex]


############################################################
# Step 0: 数据分析的初始工作, 包括读取数据文件、检查用户Id的一致性等#
############################################################
folderOfData = './data/'
# 用户登录信息
data1 = pd.read_csv(folderOfData + 'PPD_LogInfo_3_1_Training_Set.csv', header=0)

# 用户基本信息
data2 = pd.read_csv(folderOfData + 'PPD_Training_Master_GBK_3_1_Training_Set.csv', header=0, encoding='utf-8')

# 用户信息更新
data3 = pd.read_csv(folderOfData + 'PPD_Userupdate_Info_3_1_Training_Set.csv', header=0)

#############################################################################################
# Step 1: 从PPD_LogInfo_3_1_Training_Set &  PPD_Userupdate_Info_3_1_Training_Set数据中衍生特征#
#############################################################################################
# compare whether the four city variables match
data2['city_match'] = data2.apply(lambda x: int(x.UserInfo_2 == x.UserInfo_4 == x.UserInfo_8 == x.UserInfo_20), axis=1)
del data2['UserInfo_2']
del data2['UserInfo_4']
del data2['UserInfo_8']
del data2['UserInfo_20']

# 提取申请日期，计算日期差，查看日期差的分布
data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
data1['ListingGap'] = data1[['logInfo', 'Listinginfo']].apply(lambda x: (x[1] - x[0]).days, axis=1)
plt.hist(data1['ListingGap'], bins=200)
plt.title('Days between login date and listing date')
ListingGap2 = data1['ListingGap'].map(lambda x: min(x, 365))
plt.hist(ListingGap2, bins=200)

timeWindows = TimeWindowSelection(data1, 'ListingGap', range(30, 361, 30))

'''
使用180天作为最大的时间窗口计算新特征
所有可以使用的时间窗口可以有7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
在每个时间窗口内，计算总的登录次数，不同的登录方式，以及每种登录方式的平均次数
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1', 'LogInfo2']
data1GroupbyIdx = pd.DataFrame({'Idx': data1['Idx'].drop_duplicates()})

for tw in time_window:
    data1['TruncatedLogInfo'] = data1['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1.loc[data1['logInfo'] >= data1['TruncatedLogInfo']]
    for var in var_list:
        # count the frequences of LogInfo1 and LogInfo2
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_count'] = data1GroupbyIdx['Idx'].map(
            lambda x: count_stats.get(x, 0))

        # count the distinct value of LogInfo1 and LogInfo2
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1GroupbyIdx['Idx'].map(
            lambda x: uniq_stats.get(x, 0))

        # calculate the average count of each value in LogInfo1 and LogInfo2
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1GroupbyIdx[
            [str(var) + '_' + str(tw) + '_count', str(var) + '_' + str(tw) + '_unique']]. \
            apply(lambda x: DeivdedByZero(x[0], x[1]), axis=1)

data3['ListingInfo'] = data3['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
data3['UserupdateInfo'] = data3['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
data3['ListingGap'] = data3[['UserupdateInfo', 'ListingInfo']].apply(lambda x: (x[1] - x[0]).days, axis=1)
collections.Counter(data3['ListingGap'])
hist_ListingGap = np.histogram(data3['ListingGap'])
hist_ListingGap = pd.DataFrame({'Freq': hist_ListingGap[0], 'gap': hist_ListingGap[1][1:]})
hist_ListingGap['CumFreq'] = hist_ListingGap['Freq'].cumsum()
hist_ListingGap['CumPercent'] = hist_ListingGap['CumFreq'].map(lambda x: x * 1.0 / hist_ListingGap.iloc[-1]['CumFreq'])

'''
对 QQ和qQ, Idnumber和idNumber,MOBILEPHONE和PHONE 进行统一
在时间切片内，计算
 (1) 更新的频率
 (2) 每种更新对象的种类个数
 (3) 对重要信息如IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE的更新
'''
data3['UserupdateInfo1'] = data3['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx = pd.DataFrame({'Idx': data3['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3['TruncatedLogInfo'] = data3['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3.loc[data3['UserupdateInfo'] >= data3['TruncatedLogInfo']]

    # frequency of updating
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_freq'] = data3GroupbyIdx['Idx'].map(lambda x: freq_stats.get(x, 0))

    # number of updated types
    Idx_UserupdateInfo1 = temp[['Idx', 'UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3GroupbyIdx['Idx'].map(
        lambda x: uniq_stats.get(x, x))

    # average count of each type
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3GroupbyIdx[
        ['UserupdateInfo_' + str(tw) + '_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    # whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER', '_HASBUYCAR', '_MARRIAGESTATUSID', '_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3GroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3GroupbyIdx['Idx'].map(
            lambda x: item_dict.get(x, x))

# Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],
                    axis=1)
allData.to_csv(folderOfData + 'allData_0.csv', encoding='utf-8')

#######################################
# Step 2: 对类别型变量和数值型变量进行补缺#
######################################
allData = pd.read_csv(folderOfData + 'allData_0.csv', header=0, encoding='utf-8')
allFeatures = list(allData.columns)
allFeatures.remove('target')
if 'Idx' in allFeatures:
    allFeatures.remove('Idx')
allFeatures.remove('ListingInfo')

# 检查是否有常数型变量，并且检查是类别型还是数值型变量
numerical_var = []
for col in allFeatures:
    if len(set(allData[col])) == 1:
        print('delete {} from the dataset because it is a constant'.format(col))
        del allData[col]
        allFeatures.remove(col)
    else:
        # uniq_vals = list(set(allData[col]))
        # if np.nan in uniq_vals:
        # uniq_vals.remove(np.nan)
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_var.append(col)

categorical_var = [i for i in allFeatures if i not in numerical_var]

# 检查变量的最多值的占比情况,以及每个变量中占比最大的值
records_count = allData.shape[0]
col_most_values, col_large_value = {}, {}
for col in allFeatures:
    value_count = allData[col].groupby(allData[col]).count()
    col_most_values[col] = max(value_count) / records_count
    large_value = value_count[value_count == max(value_count)].index[0]
    col_large_value[col] = large_value
col_most_values_df = pd.DataFrame.from_dict(col_most_values, orient='index')
col_most_values_df.columns = ['max percent']
col_most_values_df = col_most_values_df.sort_values(by='max percent', ascending=False)
pcnt = list(col_most_values_df[:500]['max percent'])
vars = list(col_most_values_df[:500].index)
plt.bar(range(len(pcnt)), height=pcnt)
plt.title('Largest Percentage of Single Value in Each Variable')

# 计算多数值产比超过90%的字段中，少数值的坏样本率是否会显著高于多数值
large_percent_cols = list(col_most_values_df[col_most_values_df['max percent'] >= 0.9].index)
bad_rate_diff = {}
for col in large_percent_cols:
    large_value = col_large_value[col]
    temp = allData[[col, 'target']]
    temp[col] = temp.apply(lambda x: int(x[col] == large_value), axis=1)
    bad_rate = temp.groupby(col).mean()
    if bad_rate.iloc[0]['target'] == 0:
        bad_rate_diff[col] = 0
        continue
    bad_rate_diff[col] = np.log(bad_rate.iloc[0]['target'] / bad_rate.iloc[1]['target'])
bad_rate_diff_sorted = sorted(bad_rate_diff.items(), key=lambda x: x[1], reverse=True)
bad_rate_diff_sorted_values = [x[1] for x in bad_rate_diff_sorted]
plt.bar(x=range(len(bad_rate_diff_sorted_values)), height=bad_rate_diff_sorted_values)

# 由于所有的少数值的坏样本率并没有显著高于多数值，意味着这些变量可以直接剔除
for col in large_percent_cols:
    if col in numerical_var:
        numerical_var.remove(col)
    else:
        categorical_var.remove(col)
    del allData[col]

'''
对类别型变量，如果缺失超过80%, 就删除，否则当成特殊的状态
'''
missing_pcnt_threshould_1 = 0.8
for col in categorical_var:
    missingRate = MissingCategorial(allData, col)
    print('{0} has missing rate as {1}'.format(col, missingRate))
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        # In this way we convert NaN to NAN, which is a string instead of np.nan
        allData[col] = allData[col].map(lambda x: str(x).upper())

allData_bk = allData.copy()
'''
检查数值型变量
'''
missing_pcnt_threshould_2 = 0.8
deleted_var = []
for col in numerical_var:
    missingRate = MissingContinuous(allData, col)
    print('{0} has missing rate as {1}'.format(col, missingRate))
    if missingRate > missing_pcnt_threshould_2:
        deleted_var.append(col)
        print('we delete variable {} because of its high missing rate'.format(col))
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[col] == allData[col]][col]
            # makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
            missing_position = allData.loc[allData[col] != allData[col]][col].index
            not_missing_sample = random.sample(list(not_missing), len(missing_position))
            allData.loc[missing_position, col] = not_missing_sample
            # del allData[col]
            # allData[col] = makeuped
            missingRate2 = MissingContinuous(allData, col)
            print('missing rate after making up is:{}'.format(str(missingRate2)))

if deleted_var != []:
    for col in deleted_var:
        numerical_var.remove(col)
        del allData[col]

allData.to_csv(folderOfData + 'allData_1.csv', header=True, encoding='utf-8', columns=allData.columns, index=False)
