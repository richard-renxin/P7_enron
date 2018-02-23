
# coding: utf-8

# # 进阶项目3：从安然公司邮件中发现欺诈证据

# In[1]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


# ## 数据探索

# In[2]:

### 载入数据集
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict


# In[3]:

# 数据点总数
print u'数据点总数:{}'.format(len(data_dict))
print '*'*100


# In[4]:

#类之间的分配（POI/非 POI)
def count_poi(data_dict):
    n = 0
    for name in data_dict:
        if data_dict[name]["poi"] == True:
            n += 1
    return n
p = count_poi(data_dict)
print u'类之间的分配（POI/非 POI）:{}/{}'.format(p,146-p)
print '*'*100


# In[5]:

#使用的特征数量
features_list=['poi','bonus','salary','to_messages', 'deferral_payments', 

               'expenses','deferred_income', 'long_term_incentive',

               'restricted_stock_deferred', 'shared_receipt_with_poi',

               'loan_advances', 'from_messages', 'other', 'director_fees', 

               'total_stock_value', 'from_poi_to_this_person',

               'from_this_person_to_poi', 'restricted_stock',  

               'total_payments','exercised_stock_options','email_address']
print u'使用的特征数量:{}'.format(len(features_list)-1)
print '*'*100


# In[6]:

#是否有哪些特征有很多缺失值
def count_nan(data_dict, features_list):
    null_value = {}
    for name in data_dict:
        for feature in features_list:
            if data_dict[name][feature] == 'NaN':
                if feature not in null_value:
                    null_value[feature] = 1 
                else:
                    null_value[feature] += 1
    return null_value
nan_fea = count_nan(data_dict, features_list)
nan_feature = sorted(nan_fea.items(), key=lambda x: x[1],reverse=True)
print u'以下特征含有缺失值:'
for i in nan_feature:
    print i
print '*'*100


# In[7]:

#是否有哪些人有很多缺失值
def count_nan_p(data_dict, features_list):
    null_value = {}
    for name in data_dict:
        for feature in features_list:
            if data_dict[name][feature] == 'NaN':
                if name not in null_value:
                    null_value[name] = 1 
                else:
                    null_value[name] += 1
    return null_value
nan_peo = count_nan_p(data_dict, features_list)
nan_people = sorted(nan_peo.items(), key=lambda x: x[1],reverse=True)
print u'有一个人所有特征全部为NaN，没有有用信息:'
print nan_people[:1]
print '*'*100


# ## 异常值调查

# In[8]:

#确定财务数据中的异常值，并解释如何消除或以其他方式处理它们。
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
data_dict.pop('LOCKHART EUGENE E',0)
print u'删除以下异常值:\nTOTAL\nTHE TRAVEL AGENCY IN THE PARK\nLOCKHART EUGENE E'
print '*'*100


# ## 创建新特征

# In[9]:

#书面回复中提供了选择该特征的理由，并测试了该特征对最终算法性能的影响。
# 总报酬+总股票 = 总收入
for keys,features in data_dict.items():
        if features['total_payments'] == "NaN" or features['total_stock_value'] == "NaN":
            features['total_net_worth'] = "NaN"
        else:
            features['total_net_worth'] = features['total_payments'] + features['total_stock_value']
features_list += ['total_net_worth']
len(features_list)
print u'添加一个新的财务特征:total_net_worth(总报酬+总股票 = 总收入)'
print '*'*100


# In[10]:

#查看新特征有多少缺失值
def count_nan_newfeature(data_dict, features_list):
    n = 0
    for name in data_dict:
        if data_dict[name]['total_net_worth'] == 'NaN':
            n += 1
    return n
nan_newfeature = count_nan_newfeature(data_dict, features_list)
print u'新特征total_net_worth缺失值为:{}'.format(nan_newfeature)
print '*'*100


# ## 选择算法

# In[11]:

#至少尝试了 2 种不同的算法并比较了它们的性能，最终分析中使用了性能较高的一个.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



#朴素贝叶斯
NB = GaussianNB()

#决策树
dt = DecisionTreeClassifier()

#k近邻
kn=KNeighborsClassifier()


# ## 验证策略

# In[12]:

# 定义测试函数
from sklearn.cross_validation import StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score


PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


#存储到我的数据集下便于导出
my_dataset = data_dict        


# ## 测试新特征

# In[13]:

#测试新特征
print u'测试新特征对最终算法性能的影响:'
features_new = ['poi', 'total_net_worth']
print u'朴素贝叶斯:'
test_classifier(NB,my_dataset,features_new,folds = 1000)

print u'决策树:'
test_classifier(dt,my_dataset,features_new,folds = 1000)

print u'k近邻:'
test_classifier(kn,my_dataset,features_new,folds = 1000)
print '*'*100


# ## 明智地选择特征

# In[14]:

#部署单变量或递归特征
#对于支持获取特征重要性（如：决策树）或特征得分（如：SelectKBest）的算法，进行记录.
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest,f_classif

#分析财务特征时删除邮件地址,str和float冲突
features_list.remove('email_address')

#提取特征和标签
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[15]:

print u'使用SelectKBest的算法选择10个得分最高的特征:'
#获取最重要10个特征
features_selected=[]
clf = SelectKBest(f_classif, k=10)
selected_features = clf.fit_transform(features, labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_list_10 = ['poi']+features_selected


#特征得分
features_score = zip(features_list_10[1:25], clf.scores_[:24])
features_score = sorted(features_score, key=lambda s: s[1], reverse=True)
for i in features_score:
    print i
print '*'*100


# In[55]:

print u'大多数都是财务特征,只有一个邮件特征排在第七位'
print '*'*100


# ### 使用SelectKBest尝试不同的特征组合(k=10~1)，并记录了每种组合的性能

# In[52]:

#测试选择6个最佳特征,各分类器得分
features_selected=[]
clf = SelectKBest(f_classif, k=6)
selected_features = clf.fit_transform(features, labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_list_6 = ['poi']+features_selected

print u'默认参数的朴素贝叶斯分类器最高得分为使用6个最佳特征:'
print u'朴素贝叶斯:'
test_classifier(NB,my_dataset,features_list_6,folds = 1000)

print '*'*100


# In[54]:

#测试选择3个最佳特征,各分类器得分
features_selected=[]
clf = SelectKBest(f_classif, k=3)
selected_features = clf.fit_transform(features, labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_list_3 = ['poi']+features_selected

print u'默认参数的决策树分类器最高得分为使用3个最佳特征:'
print u'决策树:'
test_classifier(dt,my_dataset,features_list_3,folds = 1000)

print '*'*100
print u'默认参数的k近邻分类器得分不理想:'
print u'k近邻:'
test_classifier(kn,my_dataset,features_list_3,folds = 1000)
print '*'*100


# In[53]:

print '*'*100
print u'默认参数下,精确度和召回率最高得分是朴素贝叶斯分类器,SelectKBest(k=6),精确度0.48,召回率0.36,f1=0.41'

#特征得分
features_score = zip(features_list_6[1:25], clf.scores_[:24])
features_score = sorted(features_score, key=lambda s: s[1], reverse=True)
print u'使分类器精确度和召回率最高的六个特征为:'
for i in features_score:
    print i
print '*'*100


# ## 调整算法

# In[60]:

#调整算法
print u'调整K近邻分类器参数,提高算法性能:'
knc=KNeighborsClassifier(n_neighbors=2, weights='distance', n_jobs=-1)
print u'正在计算,预计时间为2分钟..'
test_classifier(knc,my_dataset,features_list_3,folds = 1000)


# In[65]:

print u'***************最终算法*********************'
print u'调参后的KNeighborsClassifier性能最高'
print u'参数为:  n_neighbors=2, weights=distance'
print u'Precision: 0.49715   Recall: 0.39250   F1: 0.43867'


# In[14]:

from tester import dump_classifier_and_data
dump_classifier_and_data(knc, my_dataset, features_list_3)

