import pandas as pd
import numpy as np
import sys

# 显示所有列
pd.set_option('display.max_columns', None)
# 读入接入信息
df = pd.read_csv('./data/LoanStats3a.csv', skiprows=1, low_memory=True)
# print(df.head())
# print(df.info())
print("===========")
# 数据预处理
# 删除肉眼可见的空值列
df.drop('id', axis=1, inplace=True)
df.drop('member_id', axis=1, inplace=True)

# term一列只保留数字
df['term'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

# int_rate一列只保留数字
df['int_rate'].replace(to_replace='%', value='', inplace=True)

# 计算某一列特征的种类有多少个
# print(df['emp_length'].value_counts())

df.drop('sub_grade', axis=1, inplace=True)
df.drop('emp_title', axis=1, inplace=True)

# 将emp_length转换为数字型
df['emp_length'].replace(to_replace='n/a', value=None, inplace=True)
df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

# any 当有一个空值时 就删掉一列，  all全部为空就删掉
# 删除列空值  用df.info()才会出来更多的信息
df.dropna(axis=1, how='all', inplace=True)
df.dropna(axis=0, how='all', inplace=True)

# 删除不为空，但特征重复较多的列。
# for col in df.select_dtypes(include=['float']).columns:
#    print('col {} has {}'.format(col, len(df[col].unique())))


# 删除float类型中重复较多的列
# 小于10分之1的都删掉
df.drop(['delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', \
         'total_acc', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq', \
         'chargeoff_within_12_mths', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens'], axis=1, inplace=True)

# for col in df.select_dtypes(include=['object']).columns:
#     print('col {} has {}'.format(col, len(df[col].unique())))

df.drop(['term', 'grade', 'emp_length', 'home_ownership', 'verification_status', 'pymnt_plan', \
         'issue_d', 'purpose', 'zip_code', 'addr_state', 'earliest_cr_line', 'initial_list_status', \
         'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'application_type'], axis=1, inplace=True)
# print(df.info())
df.drop(['title', 'desc'], axis=1, inplace=True)

# print(df['loan_status'].value_counts())

# 标签二值化
df['loan_status'].replace('Fully Paid', value=int(1), inplace=True)
df['loan_status'].replace('Charged Off', value=int(0), inplace=True)
df['loan_status'].replace('Does not meet the credit policy. Status:Fully Paid', np.nan, inplace=True)
df['loan_status'].replace('Does not meet the credit policy. Status:Charged Off', np.nan, inplace=True)
df['loan_status'].replace('Current', np.nan, inplace=True)
df['loan_status'].replace('Late (31-120 days)', np.nan, inplace=True)
df['loan_status'].replace('In Grace Period', np.nan, inplace=True)
df['loan_status'].replace('Late (16-30 days)', np.nan, inplace=True)
df['loan_status'].replace('Late (31-120 days)', np.nan, inplace=True)
df['loan_status'].replace('Default', np.nan, inplace=True)
df.dropna(subset=['loan_status'], how='any', inplace=True)
print(df['loan_status'].value_counts())

df.fillna(0.0, inplace=True)
# print(df.head())


print(df.info())


# 协方差矩阵,检测清洁后样本特征的相关性，去除多重相关特征（保留一列）
cor = df.corr()
cor.iloc[:, :] = np.tril(cor, k=-1)
cor = cor.stack()
print(cor[(cor > 0.55) | (cor < -0.55)])


#  删除相关系数大于0.95的
df.drop(['loan_amnt','funded_amnt','total_pymnt'],axis=1,inplace=True)

# 进行哑编码   数值不做哑编码
df = pd.get_dummies(df)
# df.to_csv('./data/feature03.csv')
