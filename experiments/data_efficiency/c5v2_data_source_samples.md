# C5-v2 data source samples (10 per source)

Random seed 42; reservoir sampling. Each sample truncated to 1500 characters.

Sources go into the C5-v2 training mix:
- Stage 1 (15.4 B trained): 80% code + 20% markup. Code = 3 sources at token-proportional weights (Stack-Edu Py ~54%, Code-Concepts ~45%, Unconditional-Algorithmic ~1%). Markup = Stack-Edu Markdown.
- Stage 2 (15.4 B trained): 90% DCLM + 10% (80% code + 20% markup, same ratios as stage 1).

---

## Stack-Edu / Python @ score > 3.0 (SWH-fetched content)  [code]

Source: `/fsx/users/dongweij/marin/outputs/raw/stack-edu-python-content/content.jsonl.gz`

### Sample 1

_(repo: jinger02/testcodes / /encodeeer.py score 3.12)_

```


s = open('encoded.txt','w')


alphabet = 'aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ'
key = 'xXzZnNlLwWeEbBgGjJhHqQdDyYvVtTkKfFuUoOmMpPcCiIaAsSrR'


secret_message = open('decoded.txt')

for item in secret_message:
    line = item
    
    for character in line:
        if character.isalpha():
            print(key[alphabet.index(character)],end='', file=s)
        else:
            print(character, end='', file=s)

    
s.close()


```

### Sample 2

_(repo: wofud39/Programming_Python / /개인/거스름돈.py score 3.48)_

```
a = input('값을 입력하시오. >> ')
a = int(a)

list1 = [500,100,50,10]

count = 0

while (True):
    print(f'{list1[count]}원: {a//int(list1[count])}')
    a = a%list1[count]
    count+=1
    if count ==int(len(list1)):
        break
```

### Sample 3

_(repo: JANGHEEEUN/Keras / /Keras/keras06_split2.py score 3.14)_

```
import numpy as np 

#1. 데이터 - 전처리를 할 필요가 없는 정제된 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,shuffle=False) 
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)

print(x_train) 
print(x_test)
print(x_val)

'''
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_shape = (1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_data=(x_val,y_val)) #1.train -> 2.val  
                                                                                    # val: 정확도를 높이는데 큰 영향
    
#4. 평가 예측
loss, mae = model.evaluate(x_test,y_test, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)


x_prd = np.array([201,202,203])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)
'''
```

### Sample 4

_(repo: tonyiovino/no_term / /thesum.py score 3.72)_

```
import os

f = 'sum.dat'
f_num = 0

if os.path.isfile(f):
	f = open(f, 'r')
	string = f.readline()
	f_num = int(string)
	print('Lettura dal file sum.dat in corso...')
	print('Valore corrente: ', f_num)
	print('Faro la somma di un nuovo valore.')

	f.close()
else:
	print('Il file sum.dat non esiste.')

num = input('Creare un nuovo valore: ')
num = int(num)
somma = f_num + num
somma = str(somma)

f = open('sum.dat', 'w')

f.write(somma)

print('Ho letto il valore ', somma)

print('Salvataggio nel file sum.dat in corso...')

f.close()

```

### Sample 5

_(repo: Anightingale/imdb-actor-search / /imdbsearch/__main__.py score 3.36)_

```
# IMPORTS
from bs4 import BeautifulSoup
import requests
import urllib.parse
from datetime import date
import json

"""
  Given a name, imbdsearch queries IMDB and returns a list of movies 
  that individual has appeared in
"""
def main():

	actordict = {}

	# make sure actor name exists
	descending = 'n'
	actorinfo = 0
	while actorinfo == 0:
		inputname = input("Please enter actor\'s name: ")
		actorinfo = getactorinfo(inputname)
	
	descending = input("List sorted in descending order? (y/n): ")

	print("\nDisplaying movies from{}".format(actorinfo.get_text()))

	# store all actor information into dictionary
	actordict['ActorName'] = actorinfo.get_text().split('(')[0][1:-1]
	actordict['KnownFor'] = actorinfo.get_text().split(',')[1][1:-1]

	actordict['Movies'] = getactormovies(actorinfo, descending)

	if input("\nExport to JSON file? (y/n): ") == 'y':
		exporttoJSON(actordict)
	


def exporttoJSON(actordict): 
	"""
	takes a dictionary of information about the actor 
	and prints it to a JSON file 
	"""
	filename = '{}.json'.format(actordict['ActorName'].replace(' ','_'))
	with open(filename, 'w') as outfile:
		json.dump(actordict, outfile)
	
	print("Exported as {}!".format(filename))
	outfile.close()

def getactormovies(actor, order): 
	"""
	Given the actor, queries IMDB and returns a dictionary of information 
	about the movies the individual has appeared in
	"""
	actorcode = actor.find('a')['href']
	#href for actor's profile page comes in format /name/actorcode/ want to extra

... [truncated; full doc has 4,232 chars]
```

### Sample 6

_(repo: Prashant414/python / /qn 31.py score 3.06)_

```
# Find the sum of the series 2 +22 + 222 + 2222 + .. n terms
a=0
sum=0
for i in range (0,5):
    a=a+(2*(10**i))
    sum=sum+a
print(sum)

```

### Sample 7

_(repo: aaroncymor/coding-bat-python-exercises / /List-1/first_last6.py score 3.78)_

```
"""
	Given an array of ints, return True if 6 appears as ether the first or last element in the array.
	The array will be length 1 or more.
"""

def first_last6(nums):
	if nums[0] == 6 or nums[len(nums) - 1] == 6:
		return True
	return False
```

### Sample 8

_(repo: MoustafaaAshraf/featurengineering / /featurengineering.py score 3.47)_

```
class FeatureEngineering():
    def __init__(self, df):
        '''
        Initialising a feature engineering instance is only used on a certain dataframe
        '''
        self.df = df

    def rare_labels(self, df, feature, threshhold):
        '''
        A function for changing the category of labels less than the threshhold as 'Other'
        @param DataFrame, feature, threshhold
        @return A rare-encoded feature df[feature]
        '''
        counts = self.df[feature].value_counts()
        # Recording the value counts of each categorical label in the feature

        common = list(counts[counts > threshhold].index)

        # Filtering the common labels based on the chosen threshhold, dicided by user.

        def rare(x):
            '''
            A sub-function for labeling the rare values based on the common ones previously dicided on.
            @param: An observation from the feature column
            @return: either 'Other', the original observation, based on either being within the common or not.
            '''
            return 'other' if x not in common else x

        self.df[feature] = self.df[feature].apply(rare)
        # overwriting the original feature column
        return self.df[feature]
```

### Sample 9

_(repo: NehaBhatt2511/assign2 / /new.py score 3.80)_

```
import turtle
import math

new = turtle.Turtle()
new.color("blue")
new.speed(100)

for i in range(100):
    new.forward(300)
    new.left(170)
    new.circle(50)
    new.forward(20)
    new.left(math.cos(i)*10)
    
    




turtle.done()
```

### Sample 10

_(repo: itsolutionscorp/AutoStyle-Clustering / /all_data/exercism_data/python/difference-of-squares/0bea4270c0d343fd856300c0f98cf3bf.py score 3.83)_

```
def square_of_sum(n):
    return sum(range(1, n+1))**2


def sum_of_squares(n):
    return sum(x**2 for x in range(1, n+1))


def difference(n):
    return square_of_sum(n) - sum_of_squares(n)


```

---

## Stack-Edu / Markdown @ score > 3.0 (SWH-fetched content)  [markup]

Source: `/fsx/users/dongweij/marin/outputs/raw/stack-edu-markdown-content/content.jsonl.gz`

### Sample 1

_(repo: k3vnb/Find-A-Doctor / /README.md score 3.08)_

```
# Find a Doctor
### This is a demo project for returning query results via BetterDoctor API utilizing asynchronous properties of JavaScript with separated back-end and front-end functionality. 1/12/17
### by **Kevin Boyle**

## Description

_Users will enter search query items into a form, which will trigger an API call to BetterDoctor API, returning results to the user based on their query inputs. It can be found at https://lemurriot.github.io/Find-A-Doctor/._


## Specs & Planning

1. User may enter a medical issue into a form & receive doctor list with corresponding info as a result (if API call is successful & query matches items provided by API).
    * Example Input: "fever"
    * Example Output:
        - "First name: Jane,
           Last name: Dough,
           Address: 123 Main St., Portland, OR 97201,
           Phone: 503-555-1234,
           Website: www.healthwebsite.com,
           Accepting New Patients?: Yes"
2. User may enter a doctor's name into a form & receive corresponding doctor info as a result (if API call is successful & query matches items provided by API).
    * Example Input: "Jane Dough"
    * Example Output:
        - "First name: Jane,
           Last name: Dough,
           Address: 123 Main St., Portland, OR 97201,
           Phone: 503-555-1234,
           Website: www.healthwebsite.com,
           Accepting New Patients?: Yes"
3. If API call is unsuccessful user will be given a message accordingly.
  * Example Input: "Jane Dough".
  * Exampl

... [truncated; full doc has 3,114 chars]
```

### Sample 2

_(repo: frictionlessdata/tableschema-java / /docs/creating-schemas.md score 3.20)_

```
# Creating Schemas

- [Creating from scratch via Java methods](#via-java-methods)
- [Creating from a serialized JSON representation](#from-json)
- [Creating from sample data (inferring)](#inferring-a-schema-from-data)
- [Schema validation](#schema-validation)
- [Writing a Schema to a File](#writing-a-schema-to-a-file)


## Via Java methods

You can build a `Schema` instance from scratch or modify an existing one:

```java
Schema schema = new Schema();

Field nameField = new StringField("name");
schema.addField(nameField);

Field coordinatesField = new GeopointField("coordinates");
schema.addField(coordinatesField);

System.out.println(schema.asJson());

// {"fields":[{"name":"name","format":"default","description":"","type":"string","title":""},{"name":"coordinates","format":"default","description":"","type":"geopoint","title":""}]}
```

## From JSON

You can also build a `Schema` instance with `JSONObject` instances instead of `Field` instances:

```java
Schema schema = new Schema(); // By default strict=false validation

JSONObject nameFieldJsonObject = new JSONObject();
nameFieldJsonObject.put("name", "name");
nameFieldJsonObject.put("type", Field.FIELD_TYPE_STRING);
schema.addField(nameFieldJsonObject);

// Because strict=false, an invalid Field definition will be included.
// The error will be logged/tracked in the error list schema.getErrors().
JSONObject invalidFieldJsonObject = new JSONObject();
invalidFieldJsonObject.put("name", "id");
invalidFieldJsonObject.put("typ

... [truncated; full doc has 5,260 chars]
```

### Sample 3

_(repo: chellking/edit_list / /README.md score 3.50)_

```
# edit_list


[toc] 目录1
[toc] 目录2

### 一、标题

还可增加二、三、四、五、六级标题，总共六级

``` 
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
``` 

** 效果：**

# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题

```
分隔
```

### 二、粗体和斜体

用两个 \*\* 包含一段文本就是粗体的语法 \*\*

用一个 \* 包含一段文本就是斜体的语法 \*

** 效果：**

用两个 ** 包含一段文本就是粗体的语法 **

用一个 * 包含一段文本就是斜体的语法 *


### 四、代码高亮

\`\`\` java
class SomeClass(){
 String message = 'interpreter';
}
\`\`\`

** 效果：**

``` java
class SomeClass(){
 String message = 'interpreter';
}
```

### 五、制作待办事项To-do List

- [x] 已经完成项目1
    - [x] 已经完成事项1
    - [x] 已经完成事项2
- [ ] 待办事项1
- [ ] 待办事项2

### 六、列表
注：-、1.和文字之间要保留一个字符的空格。

#### 无序列表

- 列表1
    - 列表1.1
    - 列表1.2
- 列表2
- 列表3

#### 有序列表

1. 列表1
  1. 列表1.1
  2. 列表1.2
1. 列表2
1. 列表3


### 七、引用

\> 记录，成为更好的自己。---有云

** 效果：**


> 记录，成为更好的自己。---有云

### 九、分割线

这是第一段内容
\*\*\*
这是第二段内容


** 效果：**

这是第一段内容
***
这是第二段内容

### 十、连接与图片

#### 插入链接

[超级连接](http://github.com/iuap3)

#### 插入图片
![这是个图片](http://)


### 三、高效绘制 流程图、序列图、甘特图、表格




#### 八、表格

|header1 | hedader2|
|---|---|
|row1 col1 | row1 col2|
|row2 col1 | row2 col2|

** github格式 **

First Header \| Second Header
------------ \| -------------
Content from cell 1 \| Content from cell 2
Content in the first column \| Content in the second column

效果：

First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column



#### 流程图：
```
graph TD
A[Christmas] -->B(Go shopping)
B --> C{Let me

... [truncated; full doc has 3,398 chars]
```

### Sample 4

_(repo: Sumiya-Akter/CSCI39548-Final-Project-Client / /README.md score 3.72)_

```
# Starter code for CRUD App

## Client 

All Campuses and Students

Frontend (React-Redux, React, and React Router)
- [ ] Write a campuses sub-reducer to manage campuses in your Redux store
- [ ] Write a students sub-reducer to manage students in your Redux store

- [x] Write a component to display a list of all campuses (just their names and images)
- [x] Write a component to display a list of all students (just their names)
- [x] Display the all-campuses component when the url matches `/campuses`
- [x] Display the all-students component when the url matches `/students`
- [x] Add links to the navbar that can be used to navigate to the all-campuses view and the all-students view

Single Student and Single Campus 

Frontend (React and React Router)
- [x] Write a component to display a single campus with the following information:
  - [x] The campus's name, image, address and description
  - [x] A list of the names of all students in that campus (or a helpful message if it doesn't have any students)
- [x] Display the appropriate campus's info when the url matches `/campuses/:campusId`
- [x] Clicking on a campus from the all-campuses view should navigate to show that campus in the single-campus view

- [x] Write a component to display a single student with the following information:
  - [x] The student's full name, email, image, and gpa
  - [x] The name of their campus (or a helpful message if they don't have one)
- [x] Display the appropriate student when the url matches `/stud

... [truncated; full doc has 3,537 chars]
```

### Sample 5

_(repo: allan-zhou/blog / /source/_posts/2017-06-27-javascript异步编程（1）什么是异步.md score 3.67)_

```
---
title: javascript异步编程（1）什么是异步
date: 2017-06-27 22:53:17
categories:
- Javascript
tags:
- Javascript
- 异步编程
---

# 开发中常见的异步操作
* 网络请求，如`ajax` `http.get`
* IO 操作，如`readFile` `readdir`
* 定时函数，如`setTimeout` `setInterval`

 <!-- more -->

# 同步vs异步

**同步**：如果在函数A返回的时候，调用者就能够得到预期结果(即拿到了预期的返回值或者看到了预期的效果)，那么这个函数就是同步的。

例如：
```javascript
Math.sqrt(2);
console.log('Hi');

```
**异步**：如果在函数A返回的时候，调用者还不能够得到预期结果，而是需要在将来通过一定的手段得到，那么这个函数就是异步的。  

例如：
```javascript
fs.readFile('foo.txt', 'utf8', function(err, data) {
    console.log(data);
});
```
正是由于JavaScript是单线程的，而异步容易实现非阻塞，所以在JavaScript中对于耗时的操作或者时间不确定的操作，使用异步就成了必然的选择。

> 我们常说“**JavaScript是单线程的**”。
>
> 所谓单线程，是指在JS引擎中负责解释和执行JavaScript代码的线程只有一个。不妨叫它**主线程**。
>
> 但是实际上还存在其他的线程。例如：处理AJAX请求的线程、处理DOM事件的线程、定时器线程、读写文件的线程(例如在Node.js中)等等。这些线程可能存在于JS引擎之内，也可能存在于JS引擎之外，在此我们不做区分。不妨叫它们**工作线程**

# 异步过程的构成要素
从上文可以看出，**异步函数**实际上很快就调用完成了。但是后面还有工作线程执行异步任务、通知主线程、主线程调用回调函数等很多步骤。我们把整个过程叫做**异步过程**。异步函数的调用在整个异步过程中，只是一小部分。 

总结一下，一个异步过程通常是这样的：

> 主线程发起一个异步请求，相应的工作线程接收请求并告知主线程已收到(异步函数返回)；主线程可以继续执行后面的代码，同时工作线程执行异步任务；工作线程完成工作后，通知主线程；主线程收到通知后，执行一定的动作(调用回调函数)。  

异步函数通常具有以下的形式：

```javascript
A(args..., callbackFn)
```
它可以叫做异步过程的发起函数，或者叫做异步任务注册函数。args是这个函数需要的参数。callbackFn也是这个函数的参数，但是它比较特殊所以单独列出来。

所以，从主线程的角度看，一个异步过程包括下面两个要素：

* 发起函数(或叫注册函数)A
* 回调函数callbackFn  

它们都是在主线程上调用的，其中注册函数用来发起异步过程，回调函数用来处理结果。

# 消息队列和事件循环

上文讲到，异步过程中，工作线程在异步操作完成后需要通知主线程。那么这个**通知机制**是怎样实现的呢？答案是利用消息队列和事件循环。

用一句话概括：

>工作线程将消息放到消息队列，主线程通过事件循环过程去取消息。

* **消息队列（task queue）**：消息队列是一个先进先出的队列，它里面存放着各种

... [truncated; full doc has 2,832 chars]
```

### Sample 6

_(repo: xiaolei565/aimto408 / /408代码汇总（自己总结的，更新至2018年，欢迎提供其他年份的）.md score 3.88)_

```
## **408代码汇总**

1. 2009年

   已知一个带有表头结点的单链表，结点结构为：

   假设该链表只给出了头指针 list。在不改变链表的前ᨀ下，请设计一个尽可能高效的算法，查找链表中倒数第 k 个位置上的结点（k 为正整数）。若查找成功，算法输出该结点的 data 域的值，并返回 1；否则，只返回 0。

   **算法思想：使用p，q两个指针，p指针先移动扫描k个指针，之后q再与p同步移动，当p指向最后一个节点时，q正好指向倒数第k个节点**

   ```c
   int SearchRearK(LNode *L, int k)
   {
       int count=0;//用来计数
       LNode *q=L->link,*p=L->link;
       while(p!=NULL)
       ｛
           if(count<k)
               count++;
       	else
               q=q->link;//当count等于开始，q和p同步向后移动
       	p=p->link; 
       ｝
       if(count<k)
           return 0;//如果链表节点个数小于k
       else
       ｛
           printf("%d",q->data);
           return 1;
       ｝
   }
   ```

   ----

2. 2010年

   设将 n（n>1）个整数存放到一维数组 R 中。试设计一个在时间和空间两方面都尽可能高效的算法。将 R 中保存的序列循环左移 p（0<p<n）个位置，即将 R 中的数据由（X0, X1, , Xn-1）变换为（Xp, Xp+1, , Xn-1, X0, X1, , Xp-1）。

   **算法思想：先将R中前p个元素逆置，再将剩下的元素逆置，最后将R中所有的元素再整体做一次逆置即可**

      ```c
   void Reverse(int R[],int l,int r)
   {
      int i,j;
      int temp;
      for(i=l,j=r;i<j;++i,--j)
      {
          temp=R[i];
          R[i]=R[j];
          R[j]=temp
      }
   }
   void RCR(int R[],int n,int p)
   {
      if(p<=0||p>=n)
          cout<<"ERROR"<<endl;
      else
      {
          Reverse(R,0,p-1);
          Reverse(R,p,n-1);
          Reverse(R,0,n-1);
      }
   }
      ```

----

3. 2011年

   一个长度为L（L≥1）的升序序列S，处在第¬L/2º个位置的数称为S 的中位数。例如，若序列S1=（11，13，15，17，19），则S1 的中位数是15，两个序列的中位数是含它们所有元素的升序序列的中位数。例如，若S2

... [truncated; full doc has 9,096 chars]
```

### Sample 7

_(repo: diasbruno/rcwe / /documentation.md score 3.50)_

```
# rcwe documentation

## first steps

add daggy and rcwe to your project.

    npm install --save daggy rcwe

## usage

### creating the types

see daggy's documentation to learn you to create your types.

[https://github.com/fantasyland/daggy](https://github.com/fantasyland/daggy)

### creating a rcwe context

rcwe consists of a single function which is responsible to create
everything needed.

create the new types is easy:

```
import daggy from 'daggy';

const Events = daggy.taggedSum('Events', {
  Increment: ['n'],
  Decrement: ['n']
});
```

now, we are ready to create the context.

```
import RCWE from 'rcwe';

const {
  Context, Consumer, Provider,
  Increment, Decrement
} = RCWE(Events, {
  Increment: (...args) => state => ({ count: state.count + 1 }),
  Decrement: (...args) => state => ({ count: state.count - 1 })
});
```

what rcwe has returned:

- Increment and Decrement: The created types reexported.
- Context: The same context returned by `React.createContext`.
- Consumer: The same consumer from the context above.
- Provider: Here is the trick.

### provider

this provider expects a `scope`, the instance of the component where the state
is. The value is assumed from the state of the scope.

From our example:

```
class App extends React.Component {
  state = { count: 0 };

  render() {
    return (
      <Provider scope={this}>
        ...
      </Provider>
    );
  }
}
```

it will provide to the consumer 2 functions:

- apply: Execute immediatelly, so not good 

... [truncated; full doc has 2,100 chars]
```

### Sample 8

_(repo: MilesYeah/ASimpleSummary-Python / /Test.测试/ddt/ddt.file_data.从配置文件中读取参数.md score 3.11)_

```
# @file_data

file_data 读取配置文件的时候，其默认编码为 utf8 ，所以有时候在使用中文配置文件的时候，会出现读取配置文件出错的问题。

如果出现这个问题，那么我们可以来到 file_data 定义的源码部分，将 with open 的编码改掉，例如 gbk 编码。


## 传递 JSON 文件
testddt.json
```json
{
  "first": [
    {
      "isRememberMe": "True",
      "password": "111111",
      "username": "root"
    },
    "200"
  ],
  "second": [
    "{'isRememberMe': True, 'password': '1111111', 'username': 'root'}",
    "406"
  ],
  "third": [
    1,
    2
  ],
  "four": "123123"
}
```
```py
from ddt import *


# 在测试类前必须首先声明使用 ddt
@ddt
class imoocTest(unittest.TestCase):

    @file_data('F:/test/config/testddt.json')
    def test_json(self, data):
        print(data)
```
运行结果
```
[{'isRememberMe': 'True', 'password': '111111', 'username': 'root'}, '200']
["{'isRememberMe': True, 'password': '1111111', 'username': 'root'}", '406']
[1, 2]
123123
```





## 传递 YAML 文件

```txt  paras.txt
abc,def
haha,hehe
heihei,happy
```

```yml  file_data1.yml
name: "www"
info: "info"
```
```yml  file_data2.yml
- name: "Robert"
  text: "handsome"
- name: "Sophia"
  text: "beautiful"
```

```py
import os
import time
import unittest
# import HTML
from selenium import webdriver
from ddt import ddt, data, unpack, file_data
import yaml


def get_paras_from_file():
    ret = []
    with open("paras.txt", 'r') as f:
        for line in f.readlines():
            paras = line.strip().split(',')
            ret.append(paras)
    return ret


@ddt
class tUnittestDdt(unittest.TestCase):

    @data(*get_paras_from_file())
    @

... [truncated; full doc has 1,932 chars]
```

### Sample 9

_(repo: charblus/OahcDocs / /qianduan/angular/ng-template和ng-container.md score 3.08)_

```
ng-template
ng-container

```html
<h5>Table开发中</h5>
<table class="table">
  <thead>
    <tr>
      <th scope="col">
        <ng-container *ngTemplateOutlet="title"></ng-container>
      </th>
      <th scope="col">
        <ng-container *ngTemplateOutlet="dataIndex; context: columns"></ng-container>
      </th>
      <th scope="col" class="line">
        <ng-container *ngTemplateOutlet="key; context: columns"></ng-container>
      </th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<br />
<ng-template>你会显示吗？？？？？？</ng-template>
<ng-template #title>
  <span>Table1</span>
</ng-template>
<ng-template #dataIndex let-address>
  <span>{{address}}!</span>
</ng-template>
<ng-template #key let-id="key">
  <span>{{id}}!</span>
</ng-template>



<h3>template</h3>
<ng-container *ngTemplateOutlet="greet"></ng-container>
<hr>
<ng-container *ngTemplateOutlet="eng; context: myContext"></ng-container>
<hr>
<ng-container *ngTemplateOutlet="svk; context: myContext"></ng-container>
<hr>
<ng-template #greet><span>Hello</span></ng-template>
<ng-template #eng let-name><span>Hello {{name}}!</span></ng-template>
<ng-template #svk let-person="localSk"><span>Ahoj {{person}}!</span></ng-template>
```

相对应
```css
.line::before {
  content: "$"
}
```


```ts

import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-zx-base-table',
  templateUrl: './zx-base-table.component.html',
  styleUrls: ['./zx-base-table.component.scss']
})
export class ZxBaseTableComponent implement

... [truncated; full doc has 6,949 chars]
```

### Sample 10

_(repo: jiangyan1224/JavaNotes / /Java泛型补充：.md score 3.20)_

```
#### Java泛型补充：

https://mp.weixin.qq.com/s/ysrehh2b7utL-Viw-Vf3yQ

```java
package syntactic_sugar;

import java.lang.reflect.InvocationHandler;
import java.util.*;

public class GenericTest {
    public <T>void test(T a){//使用T，需要在方法上/类上有所声明
        Class<T> clazzT;
    }

    //通过T确保泛型参数的一致性：要求T要是Number或者其子类型，并且dst src的T要是同一类型
    //就算dst src都是Number子类型，比如dst Float;src Integer，报错；
    //都是Float/Integer才行
    public static <T extends Number> void test1(List<T> dst, List<T> src){

    }

    public static void main(String[] args){
        List<Float> dst = new ArrayList<>();
        List<Float> src = new ArrayList<>();
        test1(dst, src);

        Integer a= 1;
        Integer b = 2;
        Integer c = 3;
        Integer d = 3;
        Integer e = 321;
        Integer f = 321;
        Long g = 3L;
        System.out.println( c == d);//true
        System.out.println(e == f);//false
        System.out.println(c == (a + b));//true
        System.out.println(c.equals(a + b));//true
        //System.out.println(g == d);错误: 不可比较的类型: Long和Integer
        System.out.println(g == (a + b));//true
        System.out.println(g.equals(a + b));//false
    }
}

```


```

---

## Nemotron-Pretraining-Specialized-v1.1 / Code-Concepts  [code]

Source: `/fsx/users/dongweij/marin/outputs/raw/nemotron_code_concepts.jsonl.gz`

### Sample 1

_(uuid: 068bb330-cf82-4ef2-be3e-50f688ca7e2a)_

```
from typing import List, Tuple

def find_prefix_with_pattern(points: List[Tuple[int, int]], pattern: str) -> int:
    """Find the smallest prefix length of an array of 2D points such that the binary
    representation of the squared Euclidean distance from the origin of the xor-sum
    of the prefix starts with a given pattern.
    Accepts list of (x, y) integers.
    If no prefix matches, returns -1.
    >>> find_prefix_with_pattern([(1, 1), (2, 0)], '10')
    1
    >>> find_prefix_with_pattern([(1, 1), (2, 0)], '101')
    2
    >>> find_prefix_with_pattern([(1, 1), (2, 0)], '111')
    -1
    """
    # cumulative xor for x and y components
    xor_x = 0
    xor_y = 0

    for idx, (x, y) in enumerate(points, start=1):
        xor_x ^= x
        xor_y ^= y
        # squared distance from origin
        dist2 = xor_x * xor_x + xor_y * xor_y
        # binary representation without the '0b' prefix
        bin_str = bin(dist2)[2:]
        if bin_str.startswith(pattern):
            return idx
    return -1
```

### Sample 2

_(uuid: 11528840-196b-453b-810b-ee93847d6d14)_

```
from typing import List

def update_csv_tokens_with_window_counts(tokens: List[str], k: int) -> None:
    """
    Given a list `tokens` of comma‑separated values, replace each element with the
    number of occurrences of that element within the following window of length
    `k` (including the element itself). The replacement must be performed in
    place and should use a sliding‑window technique with a dictionary to
    maintain counts.

    The function returns `None`; the input list is modified directly.

    >>> tokens = ["a", "b", "a", "c", "b", "a"]
    >>> update_csv_tokens_with_window_counts(tokens, 3)
    >>> tokens
    [2, 1, 1, 1, 1, 1]

    >>> tokens = ["x", "y", "y", "z"]
    >>> update_csv_tokens_with_window_counts(tokens, 1)
    >>> tokens
    [1, 1, 1, 1]

    >>> tokens = ["1", "2", "1", "3"]
    >>> update_csv_tokens_with_window_counts(tokens, 10)
    >>> tokens
    [2, 1, 2, 1]
    """
    n = len(tokens)
    if n == 0 or k <= 0:
        # nothing to do; keep the list empty or unchanged for non‑positive k
        return

    # Keep a copy of the original values – we need them for sliding the window
    original = tokens[:]                     # type: List[str]

    # Initialise the first window [0, min(k, n))
    window_end = min(k, n)
    counts: dict[str, int] = {}
    for idx in range(window_end):
        val = original[idx]
        counts[val] = counts.get(val, 0) + 1

    # Slide the window across the list, writing the result back into `tokens`
    

... [truncated; full doc has 2,105 chars]
```

### Sample 3

_(uuid: e2d1ff93-fb4c-4a79-8923-053523e3dab7)_

```
def sort_by_fib_ascii(s: str) -> str:
    """Return a string whose characters are sorted in-place based
    on the Fibonacci number of each character's ASCII value modulo 10.
    The sorting is stable: equal keys preserve original order.
    >>> sort_by_fib_ascii('cba')
    'abc'
    >>> sort_by_fib_ascii('hello')
    'eohll'
    """
    # Fibonacci numbers for indices 0‑9
    _fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    # key function: Fibonacci of (ASCII value % 10)
    def _key(ch: str) -> int:
        return _fib[ord(ch) % 10]

    # Python's sorted is stable → preserves original order for equal keys
    return ''.join(sorted(s, key=_key))
```

### Sample 4

_(uuid: 17f5fd79-ec5e-43d4-a25e-551035c741ac)_

```
from typing import Dict, List, Tuple, Set
import math

def max_component_edge_gcd(edge_str: str) -> int:
    """Find the greatest common divisor (GCD) of edge weights for each connected component of an undirected graph described by a multiline string.
    The string contains one edge per line in the form "node1-node2:weight" where weight is a positive integer.
    For every connected component, compute the GCD of all its edge weights, then return the largest of those GCDs (0 if the graph has no edges).

    >>> max_component_edge_gcd("A-B:12\\nB-C:18\\nD-E:27\\nE-F:9")
    9
    >>> max_component_edge_gcd("X-Y:5\\nY-Z:10\\nM-N:25")
    5
    >>> max_component_edge_gcd("")
    0
    """
    # ---- helper: Disjoint Set Union (Union‑Find) ---------------------------------
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        # path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # ---- parse input -------------------------------------------------------------
    if not edge_str.strip():
        return 0

    edges: List[Tuple[str, str, int]] = []          # (u, v, weight)
    for line in edge_str.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            uv, w = line.split(":")
            u, v = uv.split("-")


... [truncated; full doc has 2,329 chars]
```

### Sample 5

_(uuid: 44939628-c5cc-4ac1-857d-14568fcdf0c8)_

```
from typing import List, Tuple

def kth_string_with_lcm(strings: List[str], k: int) -> Tuple[str, int]:
    """Return the k‑th string in lexicographic order after reversing all strings,
    along with the LCM of all string lengths.

    >>> kth_string_with_lcm(['abc', 'de', 'f'], 1)
    ('cba', 6)
    >>> kth_string_with_lcm(['a', 'b', 'c'], 3)
    ('c', 1)
    >>> kth_string_with_lcm(['abcd', 'ab', 'abc'], 2)
    ('cba', 12)
    """
    # ---- helper functions -------------------------------------------------
    def gcd(a: int, b: int) -> int:
        """Greatest common divisor using Euclid's algorithm."""
        while b:
            a, b = b, a % b
        return a

    def lcm(a: int, b: int) -> int:
        """Least common multiple based on gcd."""
        return a // gcd(a, b) * b

    # ---- main logic -------------------------------------------------------
    # 1. reverse each string
    reversed_strings = [s[::-1] for s in strings]

    # 2. sort lexicographically
    sorted_rev = sorted(reversed_strings)

    # 3. pick k‑th (1‑based) element
    kth_str = sorted_rev[k - 1]

    # 4. compute LCM of original lengths
    lengths = [len(s) for s in strings]
    if not lengths:                     # empty input edge case
        overall_lcm = 0
    else:
        overall_lcm = lengths[0]
        for length in lengths[1:]:
            overall_lcm = lcm(overall_lcm, length)

    return kth_str, overall_lcm
```

### Sample 6

_(uuid: 932e540c-9dd1-4645-8625-40fc08c31696)_

```
from typing import List

def is_valid_unique_strings(strings: List[str]) -> bool:
    """ Validate a list of strings according to the following rules:
    
    The dataset is valid if:
    • each string contains only unique characters, and
    • no character is repeated across different strings.
    
    >>> is_valid_unique_strings(["abc", "def"])
    True
    >>> is_valid_unique_strings(["abc", "bcd"])
    False
    >>> is_valid_unique_strings(["abca", "def"])
    False
    """
    # Set that will hold every character we have already seen
    used_chars = set()

    for s in strings:
        # Rule 1: characters inside a single string must be unique
        if len(s) != len(set(s)):
            return False

        # Rule 2: characters must not appear in any other string
        for ch in s:
            if ch in used_chars:
                return False
            used_chars.add(ch)

    return True
```

### Sample 7

_(uuid: 6419ce48-d3db-43d7-8110-1c1f0d77e8d3)_

```
def longest_repeating_substring_length(s: str) -> int:
    """Return the length of the longest substring that appears at least twice in
    s.  The algorithm uses dynamic programming to keep a table of
    longest common prefixes between pairs of suffixes, allowing an
    in‑place search for repeated patterns without creating many
    substring objects.
    
    >>> longest_repeating_substring_length("banana")
    3
    >>> longest_repeating_substring_length("abcd")
    0
    >>> longest_repeating_substring_length("ababab")
    4
    """
    n = len(s)
    # dp[j] will hold LCP length of suffixes starting at i+1 and j+1
    dp = [0] * (n + 1)
    best = 0

    # iterate i backwards so that dp[j+1] already represents
    # LCP(s[i+1:], s[j+1:]) when we need it
    for i in range(n - 1, -1, -1):
        new_dp = [0] * (n + 1)
        for j in range(n - 1, -1, -1):
            if s[i] == s[j]:
                # extend the previous common prefix
                new_dp[j] = dp[j + 1] + 1
                # we need two different occurrences, so ignore the case i == j
                if i != j and new_dp[j] > best:
                    best = new_dp[j]
        dp = new_dp

    return best
```

### Sample 8

_(uuid: 62b14095-15fc-4cf0-a78f-ea7c25d3d2d1)_

```
from typing import List

def average_modulo_subarray_length(nums: List[int], k: int) -> float:
    """Return the average length of all contiguous subarrays whose sum is
    divisible by k. If no such subarray exists, return 0.0.

    Examples
    >>> average_modulo_subarray_length([4, 5, 0, 3, 1], 5)
    2.2
    >>> average_modulo_subarray_length([1, 2, 3, 4, 5], 3)
    2.0
    >>> average_modulo_subarray_length([1, 1, 1], 7)
    0.0
    """
    # prefix sum modulo k: pref[i] = (nums[0] + ... + nums[i]) % k
    # A subarray (l..r) has sum % k == 0  <=>  pref[r] == pref[l-1].
    # For each modulo value we need to know how many previous indices
    # had the same remainder and the sum of those indices.
    #
    # When we are at index j (0‑based) with remainder r,
    # each earlier index i with the same remainder produces a subarray
    # of length j - i.  Sum of lengths contributed by this remainder:
    #   cnt * j - sum_i
    # where cnt = number of previous occurrences, sum_i = sum of their indices.
    #
    # We keep a dictionary: remainder -> [count, sum_of_indices].
    # The "virtual" prefix before the first element has index -1 and remainder 0.

    if k == 0:                 # defensive, division by zero makes the problem ill‑posed
        return 0.0

    # initialise with the empty prefix
    rem_info = {0: [1, -1]}     # remainder 0 seen once at index -1
    total_len = 0               # sum of lengths of all qualifying subarrays
    total_cnt = 0               #

... [truncated; full doc has 2,044 chars]
```

### Sample 9

_(uuid: e00f816e-8c4f-4841-8acd-a62bedd118f4)_

```
from typing import List

def encrypt_array_recursive(data: List[int], key: int) -> List[int]:
    """Recursively encrypt a list of byte values (0–255) by XORing each element with a given key.
The function processes the array element‑by‑element using recursion and returns a new list of
encrypted byte values.

>>> encrypt_array_recursive([1, 2, 3], 3)
[2, 1, 0]
>>> encrypt_array_recursive([10, 20, 30, 40], 5)
[15, 17, 27, 45]
>>> encrypt_array_recursive([0, 0, 0], 255)
[255, 255, 255]
"""
    # Base case – empty list
    if not data:
        return []

    # Process the first element, then recursively process the rest
    first_encrypted = data[0] ^ key          # XOR the current element with the key
    rest_encrypted = encrypt_array_recursive(data[1:], key)  # recursion

    # Combine the encrypted first element with the encrypted tail
    return [first_encrypted] + rest_encrypted
```

### Sample 10

_(uuid: 7df1ff83-04a0-4b1b-9fd4-00e1fb22f3c5)_

```
from typing import List, Tuple

def max_prime_factor_path(node_values: List[int], edges: List[Tuple[int, int]]) -> int:
    """Return the maximum possible sum of distinct prime factors of node labels
    along any shortest path from node 0 to the last node in an undirected graph.
    An edge is usable only if its endpoints share a common prime factor (gcd > 1).
    Use BFS to find the minimal number of steps; among all such shortest paths,
    pick the one with the largest sum of distinct prime factors of its nodes.
    The sum for a node counts each distinct prime factor once.
    Return -1 if no valid path exists.
    >>> max_prime_factor_path([6, 10, 15], [(0, 1), (1, 2)])
    20
    >>> max_prime_factor_path([6, 10, 15, 5], [(0, 1), (1, 3), (0, 2), (2, 3)])
    18
    >>> max_prime_factor_path([5, 7], [(0, 1)])
    -1
    """
    # ---------- helper utilities ----------
    def _gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    def _prime_factor_sum(x: int) -> int:
        """sum of distinct prime factors of x"""
        s = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                s += d
                while x % d == 0:
                    x //= d
            d += 1 if d == 2 else 2   # after 2 test only odd numbers
        if x > 1:
            s += x
        return s

    n = len(node_values)
    if n == 0:
        return -1

    # pre‑compute the factor‑sum for each node
    factor_sum = [_prime_

... [truncated; full doc has 3,004 chars]
```

---

## Nemotron-Pretraining-Specialized-v1.1 / Unconditional-Algorithmic  [code]

Source: `/fsx/users/dongweij/marin/outputs/raw/nemotron_unconditional_algorithmic.jsonl.gz`

### Sample 1

_(uuid: 08a49dcc-88f0-40c6-9782-984d9037f39f)_

```
def count_bsts_with_longest_path(n: int, k: int) -> int:
    """
    Return the number of structurally unique BSTs with n distinct keys whose height (number of nodes on the longest root‑to‑leaf path) is exactly k.Result is returned modulo 10**9+7.
    """
    MOD = 10**9 + 7
    # dp[i][h] = number of BSTs with i nodes and exact height h
    dp = [[0] * (k + 2) for _ in range(n + 1)]
    dp[0][0] = 1  # empty tree has height 0

    for nodes in range(1, n + 1):
        for height in range(1, k + 1):
            total = 0
            for left in range(nodes):
                right = nodes - 1 - left
                # case 1: left height = height-1, right height < height-1
                left_h = height - 1
                for rh in range(height - 1):
                    total = (total + dp[left][left_h] * dp[right][rh]) % MOD
                # case 2: right height = height-1, left height < height-1
                right_h = height - 1
                for lh in range(height - 1):
                    total = (total + dp[left][lh] * dp[right][right_h]) % MOD
                # case 3: both heights = height-1
                total = (total + dp[left][height - 1] * dp[right][height - 1]) % MOD
            dp[nodes][height] = total

    return dp[n][k] % MOD
```

### Sample 2

_(uuid: 73683adf-3aeb-4b68-a576-e8a76dbe45ca)_

```
Problem:
You are given an array A of N integers (1 ≤ N ≤ 2·10⁵). You may increase each element A[i] to any integer B[i] such that B[i] ≥ A[i]. Changing A[i] to B[i] costs (B[i] – A[i])². Your goal is to obtain a non‑decreasing sequence B[1] ≤ B[2] ≤ … ≤ B[N] with the minimum possible total cost

    total cost = Σ (B[i] – A[i])².

Output this minimum total cost.

Input
N
A1 A2 … AN

Output
A single integer – the minimum total cost.

Constraints
1 ≤ N ≤ 2·10⁵
|Ai| ≤ 10⁹
Result fits in signed 64‑bit integer.

Example
Input
5
3 1 2 4 2

Output
9

Explanation
The component‑wise smallest feasible non‑decreasing sequence is B = [3, 3, 3, 4, 4]. The total cost is (0)² + (2)² + (1)² + (0)² + (2)² = 9, which is minimal.

Solution:
```python
import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]

    total = 0
    prev = -10**18  # effectively -infinity
    for a in A:
        b = max(a, prev)          # smallest feasible value for this position
        diff = b - a
        total += diff * diff
        prev = b
    print(total)

if __name__ == "__main__":
    solve()
```
```

### Sample 3

_(uuid: bf1c4ca4-3269-4f22-acf4-1c431d24186c)_

```
**Question**

You are given an array `arr` of `n` integers.  
Your task is to count how many positions `i` ( `1 ≤ i < n` ) satisfy `arr[i] > arr[i‑1]`.

In other words, count the number of times an element is **strictly greater** than the element immediately before it.

**Input Format**

- The first line contains a single integer `n` — the size of the array.  
- The second line contains `n` space‑separated integers `arr[0] … arr[n‑1]`.

**Output Format**

- Print a single integer — the count of indices `i` such that `arr[i] > arr[i‑1]`.

**Constraints**

- `1 ≤ n ≤ 10^5`
- `-10^9 ≤ arr[i] ≤ 10^9`

**Sample Input 1**
```
5
1 2 2 4 3
```

**Sample Output 1**
```
2
```

**Explanation**

- `arr[1] = 2 > arr[0] = 1` → count = 1  
- `arr[2] = 2` is not greater than `arr[1]`  
- `arr[3] = 4 > arr[2] = 2` → count = 2  
- `arr[4] = 3` is not greater than `arr[3]`  

Hence the answer is `2`.

**Sample Input 2**
```
3
5 4 3
```

**Sample Output 2**
```
0
```

---

**Explanation**

Iterate through the array once, compare each element with its predecessor, and increment a counter when the current element is larger. This runs in `O(n)` time and `O(1)` extra space, which easily satisfies the constraints.

---

**Solution**

```python
import sys

def count_increasing_pairs(arr):
    """Return the number of indices i where arr[i] > arr[i-1]."""
    cnt = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            cnt += 1
    return cnt

def main():
    data = sys.stdin.rea

... [truncated; full doc has 1,876 chars]
```

### Sample 4

_(uuid: 29477332-a13b-48dd-a882-bdd8adb3fe74)_

```
import numpy as np

def sum_neighborhood(arr: np.ndarray) -> np.ndarray:
    """
    Return a new 2D array where each element is the sum of its 3x3 neighbourhood in the input array. Edge and corner positions use the available smaller neighbourhoods.
    """
    # Create an output array with the same shape and dtype as the input
    result = np.zeros_like(arr)
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            # Determine neighbourhood bounds, clamped to array edges
            r0 = max(i - 1, 0)
            r1 = min(i + 2, rows)
            c0 = max(j - 1, 0)
            c1 = min(j + 2, cols)
            # Sum the neighbourhood and store the result
            result[i, j] = np.sum(arr[r0:r1, c0:c1])
    return result
```

### Sample 5

_(uuid: ac22f517-d00c-4f3f-bee9-b05b48d8e070)_

```
**<Question>**

**Title:** Count Pairs with a Given Sum  

**Problem Statement:**  
Given an array of integers `nums` and an integer `target`, return the number of *unique* pairs `(i, j)` such that `i < j` and `nums[i] + nums[j] == target`.

**Input Format:**  
- The first line contains two space‑separated integers `n` (the size of the array) and `target`.  
- The second line contains `n` space‑separated integers representing the array `nums`.

**Output Format:**  
- Print a single integer – the number of unique pairs whose sum equals `target`.

**Constraints:**  
- `1 ≤ n ≤ 10^5`  
- `-10^9 ≤ nums[i] ≤ 10^9`  
- `-10^9 ≤ target ≤ 10^9`  

**Sample Input 1**
```
6 7
1 5 3 4 2 3
```

**Sample Output 1**
```
3
```

**Explanation:**  
The pairs are `(1,6) → 1+6=7`, `(2,4) → 5+2=7`, and `(3,5) → 3+4=7`.  
(Note: indices are 1‑based in this explanation; the algorithm works with 0‑based indices.)

**Sample Input 2**
```
5 10
5 5 5 5 5
```

**Sample Output 2**
```
10
```

**Explanation:**  
All `C(5,2) = 10` pairs sum to `10`.

---

**<Explanation>**

To count pairs efficiently we can use a hash map (`defaultdict(int)`) that stores how many times each number has been seen while iterating through the array.

For each element `x`:
1. Compute its complement `y = target - x`.
2. If `y` already exists in the map, every previous occurrence of `y` forms a valid pair with the current `x`. Add the frequency of `y` to the answer.
3. Increment the frequency of `x` in the map.

Because we only 

... [truncated; full doc has 2,614 chars]
```

### Sample 6

_(uuid: 4ec90c03-8233-4e55-b579-1d3e7cdff567)_

```
Problem:
You are given an integer array `A` of length `N` (1 ≤ N ≤ 10⁵) and two integers `K` and `L` (1 ≤ K ≤ 50, 1 ≤ L ≤ N). Select **exactly** `K` non‑overlapping subarrays from `A` such that each chosen subarray has length **at least** `L`. The sum of a chosen subarray is the sum of its elements.

Return the maximum possible total sum of the `K` selected subarrays. If it is impossible to choose `K` subarrays satisfying the constraints, output `-1`.

### Input
```
N K L
A1 A2 … AN
```
* `N K L` – three integers as described above.
* `A1 … AN` – the array elements (|Ai| ≤ 10⁹).

### Output
A single integer – the maximum total sum, or `-1` if the selection is impossible.

### Example
**Input**
```
8 2 3
1 2 -1 2 3 -2 4 5
```
**Output**
```
14
```
**Explanation**
One optimal choice is subarray `[1,2,-1,2,3]` (positions 1‑5, sum = 7) and subarray `[-2,4,5]` (positions 6‑8, sum = 7). Both have length ≥ 3, they do not overlap, and the total sum is `7 + 7 = 14`.

### Constraints
* 1 ≤ N ≤ 10⁵
* 1 ≤ K ≤ 50
* 1 ≤ L ≤ N
* |Ai| ≤ 10⁹

The answer fits into a signed 64‑bit integer.

---
## Explanation / Reasoning
We use dynamic programming. Let `pref[i]` be the prefix sum of the first `i` elements (`pref[0]=0`).
Define `dpPrev[i]` as the maximum total sum achievable using exactly `j‑1` subarrays within the first `i` elements (for the current outer loop value `j`).
For the current number of subarrays `j` we compute `dpCur[i]` – the best total using exactly `j` subarrays in the first `i` 

... [truncated; full doc has 3,868 chars]
```

### Sample 7

_(uuid: 183e98df-636a-425a-b77f-14bfb19710c2)_

```
Problem: *Divide the Array – Minimum Sum of Squares*

You are given an array `A` of length `n` (`1 ≤ n ≤ 10^5`) and an integer `m` (`1 ≤ m ≤ 500`).
You have to split the array into **exactly** `m` non‑empty contiguous sub‑arrays.
For a sub‑array `A[l … r]` let
```
S(l, r) = A[l] + A[l+1] + … + A[r]          (the sum of the sub‑array)
```
The cost of this sub‑array is `S(l, r)²`.
The total cost of a partition is the sum of the costs of its `m` parts.
Return the minimum possible total cost.
If it is impossible to split the array into `m` non‑empty parts (i.e. `m > n`) output `-1`.

**Input format**
```
n m
A1 A2 … An
```
* `n` – length of the array
* `m` – required number of parts
* `Ai` – integer values, `|Ai| ≤ 10^4`

**Output format**
```
minimum total cost
```
If no valid partition exists output `-1`.

**Example**
```
Input
5 3
1 2 3 4 5

Output
77
```
*Explanation* – One optimal partition is `[1,2,3] , [4] , [5]`:
```
(1+2+3)² = 6² = 36
4² = 16
5² = 25
Total = 36 + 16 + 25 = 77
```
The answer fits into a signed 64‑bit integer.

---

**Solution (C++17)**
```cpp
#include <bits/stdc++.h>
using namespace std;

using int64 = long long;
const int64 INF = (int64)4e18;

struct Line {
    int64 m, b;               // y = m * x + b
    Line(int64 _m = 0, int64 _b = INF) : m(_m), b(_b) {}
    int64 get(int64 x) const { return m * x + b; }
};

struct LiChao {
    struct Node {
        Line line;
        int l = -1, r = -1;
    };
    vector<Node> nodes;
    int64 X_MIN, X_MAX;
    int

... [truncated; full doc has 5,300 chars]
```

### Sample 8

_(uuid: c6785122-14c8-4dc3-bef7-d3fb92c3fce9)_

```
Problem:
Maximum Product After Decrement
You are given an integer array `nums` of length `n` (n ≥ 2). Choose two **different** indices i and j (i ≠ j). For the chosen pair compute
```
(nums[i] - 1) * (nums[j] - 1)
```
Return the maximum possible value of this expression.

Input format
- The first line contains a single integer `n` – the size of the array.
- The second line contains `n` space‑separated integers `nums[0] … nums[n‑1]`.

Output format
- Print a single integer – the maximum product after the decrement.

Constraints
- 2 ≤ n ≤ 10^5
- 1 ≤ nums[i] ≤ 10^5

Example
```
Input:
5
3 4 5 2 6

Output:
20
```
Explanation
The two largest numbers are `6` and `5`. After decrement they become `5` and `4`; their product is `5 × 4 = 20`, which is the maximum possible.

Solution Explanation:
To maximise `(a-1)*(b-1)` we need the two largest values in the original array, because subtracting `1` from each does not change the ordering. Therefore the task reduces to finding the two greatest elements of `nums`. A single linear scan keeps track of the largest (`max1`) and the second largest (`max2`). Finally return `(max1‑1)*(max2‑1)`. The algorithm runs in O(n) time and O(1) extra space.

Solution (Python 3):
```python
import sys

def max_product_after_decrement(nums):
    # initialise with very small numbers
    max1 = max2 = -float('inf')
    for x in nums:
        if x > max1:
            max2 = max1
            max1 = x
        elif x > max2:
            max2 = x
    # after the loop

... [truncated; full doc has 1,826 chars]
```

### Sample 9

_(uuid: 1e548780-0bfe-48c4-894b-6748aaeaf2d3)_

```
**Problem – Path Queries with Value Ranges**  

You are given a tree with `N` vertices (numbered `1 … N`).  
Each vertex `i` has an integer value `a[i]`.  

You have to answer `Q` queries.  
Each query consists of four integers `u, v, L, R`.  
For the simple (undirected) path that connects `u` and `v` (including both ends) you must output the number of vertices on that path whose value lies in the inclusive interval `[L, R]`.

---

### Input  

```
N Q
a[1] a[2] … a[N]
u1 v1
u2 v2
…
u(N‑1) v(N‑1)          // N‑1 edges of the tree
u1 v1 L1 R1
u2 v2 L2 R2
…
uQ vQ LQ RQ
```

* `1 ≤ N, Q ≤ 2·10⁵`
* `|a[i]| ≤ 10⁹`
* `1 ≤ ui, vi ≤ N`
* `-10⁹ ≤ L ≤ R ≤ 10⁹`

The tree is connected, there are no multiple edges or self‑loops.

### Output  

For each query output a single line containing the answer.

### Example  

```
Input
7 4
5 1 4 2 3 6 7
1 2
1 3
2 4
2 5
3 6
3 7
4 7 2 5
5 6 1 3
1 3 4 7
2 2 0 10

Output
3
2
3
1
```

*Explanation* (first query): the path `4‑2‑1‑3‑7` contains values `{2,1,5,4,7}`.  
Values in `[2,5]` are `{2,5,4}` → answer `3`.

---

## Explanation / Reasoning  

The naive way (walk the path for every query) is `O(N·Q)` – far too slow.  
We need a data structure that can answer “how many numbers in a multiset belong to `[L,R]`” quickly while we move along a path.

The classic tool for path queries on a tree is **Heavy‑Light Decomposition (HLD)**.  
HLD breaks any root‑to‑node path into `O(log N)` *segments* that are contiguous in the *base array* (the order in which no

... [truncated; full doc has 7,300 chars]
```

### Sample 10

_(uuid: f4e236d4-9da1-4fb6-a2fd-c4155e926206)_

```
"""
Write a function minimum_swaps_to_sort_circular that returns the minimum number of adjacent swaps required to sort a list of integers in non‑decreasing order, where the list may be rotated arbitrarily at no cost.
assert minimum_swaps_to_sort_circular([1, 2, 3]) == 0
"""
from typing import List
from collections import defaultdict, deque

def minimum_swaps_to_sort_circular(arr: List[int]) -> int:
    
      """Minimum adjacent swaps to sort *arr* when rotations are free.
  
      The function builds a permutation that maps each element of *arr* to the
      index it would occupy in a sorted version of the list (handling duplicates).
      It then evaluates all *n* rotations of the sorted target, computing the
      inversion count for each via a Fenwick tree, and returns the smallest
      count, which equals the minimal number of adjacent swaps required.
      """
      n = len(arr)
      if n <= 1:
          return 0
  
      sorted_arr = sorted(arr)
      # Map each value to the queue of its positions in the sorted array.
      pos_map = defaultdict(deque)
      for i, v in enumerate(sorted_arr):
          pos_map[v].append(i)
  
      # Base permutation for rotation 0.
      base = [pos_map[v].popleft() for v in arr]
      best = _inversion_count(base)
  
      cur = base
      for _ in range(1, n):
          cur = [(x + 1) % n for x in cur]
          best = min(best, _inversion_count(cur))
  
      return best

"""
Write a function highest_consonant_value that returns 

... [truncated; full doc has 3,245 chars]
```

---