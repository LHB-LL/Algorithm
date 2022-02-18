# STL容器：



## 序列式容器：



### array：

### string：

```c++
string s;
char ch;
s.push_back(ch);
```



### vector：

```c++
vector<int>s;
int x;
s.push_back(x);
for(auto i:s)
{
    cout<<i<<'\n';
}
cout<<'\n';
```



### list：

```c++
list<int>l;
list<int>::iterator pos[bbn]; 
l.push_back(1);
l.psuh_front(1);
pos[1]=l.begin();
pos[i]=l.insert(pos[x],i);
pos[i]=l.insert(next(pos[x]),i); 
for(auto i:l)
{
	cout<<i<<' ';
}
cout<<'\n';
```



### deque：



## 容器适配器：

### stack：

### queue：

### priority _ queue：

```c++
priority _ queue < int>q;//默认less< int >
降序//priority _ queue < int,vector< int >, less < int > >  q; 
升序//priority _ queue < int,vector< int >, greater< int > > q;
q.empty();
q.pop();
//自定义:
priority _ queue<A>q;
priority _ queue<A, vector<A>, B> q ; 
struct A
{
  int a,b,t;
  bool operator<(const A& y) const
  {
​    return t<y.t;
  }
};
struct B
{
  bool operator()(A x,A y)const
  {
​    return x.t<y.t;
  }
};
```



## 关联式容器：

### 红黑树结构：



#### map：

```c++
map<int,int>s;
for(auto i:s)
{
    cout<<i.first<<' '<<i.second<<endl;
}
```

#### set：

```c++
set<int>s;
int x;
s.insert(x);
for(auto i:s)
{
    cout<<i<<'\n';
}
cout<<'\n';
```



### 哈希结构：

#### unordered _ map：

```c++
unordered_map<int,int>s;
```

## Others：

### bitset:

```c++
bitset<number>s;//
for(int i=0;i<number;i++)
{
    cout<<s[i]<<' ';
}
cout<<'\n';

```

![image-20220215125439225](C:\Users\LHB\AppData\Roaming\Typora\typora-user-images\image-20220215125439225.png)

# STL算法:

### lower_bound():

二分查找>=x

### upper_bound():

二分查找>x

### next_permutation()：

```c++
vector<int>s= {3,2,1};
//string s="321";
next_permutation(s.begin(),s.end());
for(auto i:s)
{
	cout<<i<<' ';
}
cout<<'\n';
```

