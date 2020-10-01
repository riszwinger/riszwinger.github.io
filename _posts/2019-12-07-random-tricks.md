---
title: 'Random Tricks'
date: 2019-12-07
permalink: /posts/2019/12/random-tricks/
excerpt: Random tricks are a reminder to myself how I solved the problem at the time
tags:
  - Python
header:
  teaser: "/images/2019-12-07-random-tricks_files/python-img.jpg"
---

<div class="alert alert-block alert-info">
<b>Note:</b> Below tricks are a reminder to myslefy how I solved the problem at the time . . .
</div>

<img src="/images/2019-12-07-random-tricks_files/python-img.jpg">


<h1 id="tocheading">Table of Contents</h1>
<div id="toc"></div>

1. [Find words with forward slash](#Find_words_with_forward_slash)
2. [Pie & Count Plot](#Pie_Count_plot)
3. [Stacked & Parallel Bar Plot](#Stacked_parallel_bar)
4. [Boxplots](#Boxplots)
5. [Numerical Data EDA](#numerical_eda)
6. [Generate Sample DataFrame](#Sample_data)
7. [args & kwargs](#FunctionArguments)
8. [Zip unzip](#zip)




### Find words with forward slash  <a class="anchor" id="Find_words_with_forward_slash"></a>



```python
string_val="my name is a/c what can u do?"
string_val

```




    'my name is a/c what can u do?'



Start with finding Slash & space after the slash


```python
string_val.find(" ",string_val.find("/"))
```




    14




```python
string_val[:string_val.find(" ",string_val.find("/"))]
```




    'my name is a/c'



find the space before the slash by rfind


```python
string_val[:string_val.find(" ",string_val.find("/"))].rfind(" ")
```




    10




```python
string_val[:string_val.find(" ",string_val.find("/"))]\
[string_val[:string_val.find(" ",string_val.find("/"))].rfind(" "):].strip()
```




    'a/c'



## Tricks-2


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

[Data Source](https://www.kaggle.com/spscientist/students-performance-in-exams)


```python
df=pd.read_csv('StudentsPerformance.csv')
df.shape
```




    (1000, 8)




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
missing_values=df.isnull().sum()
missing_values=missing_values[missing_values>0]
missing_values
```




    Series([], dtype: int64)




```python
df.dtypes
```




    gender                         object
    race/ethnicity                 object
    parental level of education    object
    lunch                          object
    test preparation course        object
    math score                      int64
    reading score                   int64
    writing score                   int64
    dtype: object



Depending upon the data type, a different type of visualization is possible

| DataType | visualizations |
| --- | --- |
| Object | Pie , Barchart, Countplot |  
| Numerical | Boxplots, Histograms | 



```python
df['gender'].value_counts().reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>518</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>482</td>
    </tr>
  </tbody>
</table>
</div>



### Pie & Count Plot <a class="anchor" id="Pie_Count_plot"></a>



```python
fig,ax=plt.subplots(1,2,figsize=(12,5))
labels = ['Female', 'Male']
colors = ['pink', 'lightblue']
explode = (0, 0.1)
ax[0].pie(df.gender.value_counts(), startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%',
        explode=explode, shadow=True,labels=labels,)
ax[0].set_title('Gender Distribution - Pie Chart')

sns.countplot(x=df.gender,ax=ax[1],palette=colors)
for idx, row in pd.DataFrame(df['gender'].value_counts().reset_index()).iterrows():
    ax[1].text(idx,row.gender, str(round(row.gender)),color='black', ha="center")
ax[1].set_title("Count Plot")
plt.tight_layout()
plt.show()
```


![png](/images/2019-12-07-random-tricks_files/2019-12-07-random-tricks_22_0.png)



```python
d1=pd.crosstab(df["parental level of education"],columns=df.gender,margins=True)
d1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>gender</th>
      <th>female</th>
      <th>male</th>
      <th>All</th>
    </tr>
    <tr>
      <th>parental level of education</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>associate's degree</th>
      <td>116</td>
      <td>106</td>
      <td>222</td>
    </tr>
    <tr>
      <th>bachelor's degree</th>
      <td>63</td>
      <td>55</td>
      <td>118</td>
    </tr>
    <tr>
      <th>high school</th>
      <td>94</td>
      <td>102</td>
      <td>196</td>
    </tr>
    <tr>
      <th>master's degree</th>
      <td>36</td>
      <td>23</td>
      <td>59</td>
    </tr>
    <tr>
      <th>some college</th>
      <td>118</td>
      <td>108</td>
      <td>226</td>
    </tr>
    <tr>
      <th>some high school</th>
      <td>91</td>
      <td>88</td>
      <td>179</td>
    </tr>
    <tr>
      <th>All</th>
      <td>518</td>
      <td>482</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>




```python
d1.sort_values(by='All',ascending=False,inplace=True)
d2=d1.iloc[1:].drop(['All'],axis=1)
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>gender</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>parental level of education</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>some college</th>
      <td>118</td>
      <td>108</td>
    </tr>
    <tr>
      <th>associate's degree</th>
      <td>116</td>
      <td>106</td>
    </tr>
    <tr>
      <th>high school</th>
      <td>94</td>
      <td>102</td>
    </tr>
    <tr>
      <th>some high school</th>
      <td>91</td>
      <td>88</td>
    </tr>
    <tr>
      <th>bachelor's degree</th>
      <td>63</td>
      <td>55</td>
    </tr>
    <tr>
      <th>master's degree</th>
      <td>36</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



### Stacked & Parallel Bar Plot <a class="anchor" id="Stacked_parallel_bar"></a>



```python
fig,ax=plt.subplots(1,2,figsize=(12,5))
color_val=['pink','lightblue']
## Parallel bar Chart
d2.plot(kind='bar',ax=ax[0],color=color_val)
ax[0].set_xlabel('Number Of Students')

ax[0].set_title('Parallel Bar charts')

## Stacked bar Chart
d2.plot(kind='bar',ax=ax[1],stacked=True,color=color_val)
ax[1].set_xlabel('Number Of Students')

ax[1].set_title('Stacked Bar charts')
plt.suptitle('Stacked-Parallel bar Charts')
plt.show()
```


![png](/images/2019-12-07-random-tricks_files/2019-12-07-random-tricks_26_0.png)


### Boxplots <a class="anchor" id="Boxplots"></a>

To compare boxplots for columns ,convert them into rows by pandas melt and use it as below


```python
sns.boxplot(x='variable',\
            y='value',\
            data=pd.melt(df[['math score', 'reading score','writing score']],value_vars=['math score', 'reading score','writing score']),\
           palette='Set2')
plt.show()
```


![png](/images/2019-12-07-random-tricks_files/2019-12-07-random-tricks_28_0.png)



```python
d2=pd.melt(df[["parental level of education",'math score', 'reading score','writing score']],id_vars=["parental level of education"],\
        value_vars=['math score', 'reading score','writing score'])
d2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parental level of education</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bachelor's degree</td>
      <td>math score</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>some college</td>
      <td>math score</td>
      <td>69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>master's degree</td>
      <td>math score</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>associate's degree</td>
      <td>math score</td>
      <td>47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>some college</td>
      <td>math score</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,7))
sns.boxplot(x='parental level of education',\
            y='value',\
            data=d2,\
            hue='variable',\
           palette='Set2')
plt.show()
```


![png](/images/2019-12-07-random-tricks_files/2019-12-07-random-tricks_30_0.png)



```python
num_cols=df.select_dtypes(include=['int64']).columns
ctg_cols=df.select_dtypes(include=['object']).columns

print('Numerical Cols=',num_cols)
print('Categorical Cols=',ctg_cols)
```

    Numerical Cols= Index(['math score', 'reading score', 'writing score'], dtype='object')
    Categorical Cols= Index(['gender', 'race/ethnicity', 'parental level of education', 'lunch',
           'test preparation course'],
          dtype='object')
    

### Numerical Data EDA <a class="anchor" id="numerical_eda"></a>



```python
cols_val=2
fig, ax = plt.subplots(len(num_cols),cols_val,figsize=(12, 5))
colours_val=['c','brown','olive','g','y','p','m']
did_not_ran=True
for i,col in enumerate(num_cols):
    for j in range(cols_val):
        if did_not_ran==True:
            sns.boxplot(df[col],ax=ax[i,j],color=colours_val[i+j])
            ax[i,j].set_title(col)
            did_not_ran=False
        else:
            sns.distplot(df[col],ax=ax[i,j],color=colours_val[i+j])
            ax[i,j].set_title(col)
            did_not_ran=True
            
            
plt.suptitle("EDA")
plt.tight_layout()
plt.show()
```


![png](/images/2019-12-07-random-tricks_files/2019-12-07-random-tricks_33_0.png)


### Generate Sample DataFrame <a class="anchor" id="Sample_data"></a>


Below package can be used to generate sample DataFrame.
This is just awesome, wow , I've wasted so many hours building sample data and this just works so well

It generates a random data set with 30 rows and 4 columns


```python
import pandas as pd
pd.util.testing.makeDataFrame()

```

    D:\Anaconda\lib\site-packages\pandas\util\__init__.py:12: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2RRJnz2BKY</th>
      <td>0.694925</td>
      <td>1.449184</td>
      <td>-0.052772</td>
      <td>-0.517751</td>
    </tr>
    <tr>
      <th>etS4aLbtbR</th>
      <td>0.011481</td>
      <td>0.747344</td>
      <td>-0.200567</td>
      <td>-0.778930</td>
    </tr>
    <tr>
      <th>ouA6RuAKaV</th>
      <td>-0.047707</td>
      <td>-0.087963</td>
      <td>1.788137</td>
      <td>-1.011278</td>
    </tr>
    <tr>
      <th>fz9fRlOmBX</th>
      <td>1.601809</td>
      <td>-0.035541</td>
      <td>-1.103511</td>
      <td>-1.357392</td>
    </tr>
    <tr>
      <th>9LnXmw4Tni</th>
      <td>0.756518</td>
      <td>-0.409933</td>
      <td>2.830823</td>
      <td>2.969109</td>
    </tr>
    <tr>
      <th>MEj9XmaBEN</th>
      <td>-0.174688</td>
      <td>0.974406</td>
      <td>0.758869</td>
      <td>-2.270037</td>
    </tr>
    <tr>
      <th>RGbM7zeYFo</th>
      <td>0.598896</td>
      <td>-1.929452</td>
      <td>1.496389</td>
      <td>1.076925</td>
    </tr>
    <tr>
      <th>s2n8E4zYbq</th>
      <td>-2.480790</td>
      <td>-0.188385</td>
      <td>0.864637</td>
      <td>-0.539439</td>
    </tr>
    <tr>
      <th>J7AV7WGz86</th>
      <td>-1.233312</td>
      <td>-0.715990</td>
      <td>0.258828</td>
      <td>0.591111</td>
    </tr>
    <tr>
      <th>Fmsll62iRv</th>
      <td>0.775692</td>
      <td>0.386384</td>
      <td>-1.309332</td>
      <td>-2.037389</td>
    </tr>
    <tr>
      <th>kC6IdAa0pe</th>
      <td>-0.579191</td>
      <td>-0.489663</td>
      <td>0.218062</td>
      <td>0.367491</td>
    </tr>
    <tr>
      <th>yy8a0kKRBm</th>
      <td>-1.351233</td>
      <td>-0.291512</td>
      <td>0.770099</td>
      <td>-0.298295</td>
    </tr>
    <tr>
      <th>JzBFp6LyGo</th>
      <td>0.556733</td>
      <td>-1.149342</td>
      <td>-0.328176</td>
      <td>-0.782483</td>
    </tr>
    <tr>
      <th>ZUj6MtNnrQ</th>
      <td>1.198531</td>
      <td>-0.322790</td>
      <td>2.273961</td>
      <td>-0.024317</td>
    </tr>
    <tr>
      <th>Vhcf8jYUwm</th>
      <td>-0.526784</td>
      <td>0.185350</td>
      <td>-1.058503</td>
      <td>0.539276</td>
    </tr>
    <tr>
      <th>2WFULvKiU1</th>
      <td>0.991055</td>
      <td>0.340012</td>
      <td>0.240170</td>
      <td>1.007916</td>
    </tr>
    <tr>
      <th>X9z1L6HUn9</th>
      <td>-0.252113</td>
      <td>0.475670</td>
      <td>-1.228723</td>
      <td>-1.116394</td>
    </tr>
    <tr>
      <th>YISxDqK1iR</th>
      <td>0.423804</td>
      <td>-1.531481</td>
      <td>0.161626</td>
      <td>-1.327790</td>
    </tr>
    <tr>
      <th>rzyxOZnWOi</th>
      <td>1.018998</td>
      <td>0.401313</td>
      <td>-0.187255</td>
      <td>-0.877379</td>
    </tr>
    <tr>
      <th>hqGkD3y0Ht</th>
      <td>-0.531509</td>
      <td>-2.345542</td>
      <td>-0.564816</td>
      <td>1.024513</td>
    </tr>
    <tr>
      <th>SuBEchpncr</th>
      <td>-0.334761</td>
      <td>1.645064</td>
      <td>-2.448858</td>
      <td>-0.330884</td>
    </tr>
    <tr>
      <th>jsKSdPtFlM</th>
      <td>-1.357336</td>
      <td>0.296029</td>
      <td>0.039317</td>
      <td>1.144675</td>
    </tr>
    <tr>
      <th>S8D95JiuR4</th>
      <td>0.545082</td>
      <td>-0.594294</td>
      <td>-0.404269</td>
      <td>-0.637023</td>
    </tr>
    <tr>
      <th>SGQKTvOJyl</th>
      <td>-0.304923</td>
      <td>0.779594</td>
      <td>0.772719</td>
      <td>0.531959</td>
    </tr>
    <tr>
      <th>1oIga6lP2r</th>
      <td>-0.202963</td>
      <td>1.831511</td>
      <td>-0.050765</td>
      <td>0.425886</td>
    </tr>
    <tr>
      <th>P5xnWTq5nx</th>
      <td>1.474409</td>
      <td>1.228983</td>
      <td>1.331954</td>
      <td>-0.721734</td>
    </tr>
    <tr>
      <th>8jFCWjZM2c</th>
      <td>-2.559470</td>
      <td>-0.894872</td>
      <td>-1.109342</td>
      <td>1.573845</td>
    </tr>
    <tr>
      <th>4hV8WFbXg7</th>
      <td>-2.038775</td>
      <td>-0.697973</td>
      <td>1.123976</td>
      <td>0.774618</td>
    </tr>
    <tr>
      <th>hJZRiVWHO6</th>
      <td>-0.569268</td>
      <td>0.550622</td>
      <td>-0.356986</td>
      <td>-0.447904</td>
    </tr>
    <tr>
      <th>tg2WRbhqYP</th>
      <td>-1.530619</td>
      <td>-0.549201</td>
      <td>-0.521094</td>
      <td>-0.922401</td>
    </tr>
  </tbody>
</table>
</div>



### args & kwargs <a class="anchor" id="FunctionArguments"></a>


[kaggle Source](https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners)


```python
def f1(*args):
    """ *args can be one or more"""
    for val in args:
        print(val)
```


```python
f1(1,2,3,4)
```

    1
    2
    3
    4
    


```python
def f2(**kwargs):
    """ **kwargs is a dictionary"""
    for key,val in kwargs.items():
        print(key,val) 
```


```python
f2(country='india',capital='delhi',covid_cases=10)
```

    country india
    capital delhi
    covid_cases 10
    

### Zip unzip <a class="anchor" id="zip"></a>


[kaggle Source](https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners)


```python
l1=[1,2,3]
l2=[10,12,13]
list(zip(l1,l2))
```




    [(1, 10), (2, 12), (3, 13)]




```python
for i,j in zip(l1,l2):
    print(i,j)
```

    1 10
    2 12
    3 13
    

**Unzip**


```python
z1=zip(l1,l2)
s1,s2=list(zip(*z1))
print(s1,s2)
```

    (1, 2, 3) (10, 12, 13)
    


```python

```
