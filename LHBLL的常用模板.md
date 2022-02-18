# 算法模板：



## 基础：

### 归并排序———逆序对

```c++
//求逆序对数目

LL merge_sort_sum=0;
int merge_sort_a[bbn];
int merge_sort_b[bbn];
void merge_sort(int l,int r)
{
    if(l==r)
    {
        return;
    }
    int mid=(l+r)/2;

    merge_sort(l,mid);
    merge_sort(mid+1,r);

    int i=l,j=mid+1,k=l;
    while(i<=mid&&j<=r)
    {
        if(merge_sort_a[i]<=merge_sort_a[j])
        {
            merge_sort_b[k++]=merge_sort_a[i++];
        }
        else
        {
            merge_sort_b[k++]=merge_sort_a[j++];
            merge_sort_sum+=mid-i+1;
            merge_sort_sum%=mod;//看情况修改
        }
    }
    while(i<=mid)
    {
        merge_sort_b[k++]=merge_sort_a[i++];
    }
    while(j<=r)
    {
        merge_sort_b[k++]=merge_sort_a[j++];
    }
    while(l<=r)
    {
        merge_sort_a[l]=merge_sort_b[l];
        l++;
    }
}
void work()
{
    init();
    merge_sort(1,n);
    cout<<merge_sort_sum<<'\n';
}
```



## 数论：

### gcd&&lcm:

```c++
LL gcd(LL a,LL b)
{
    return b==0?a:gcd(b,a%b);
}
LL lcm(LL a,LL b)
{
    return a/gcd(a,b)*b;
}
```

### 质数判断：

```c++
bool is_prime(LL x)
{
	if(x==1||x==2) return 1;
	if(x%2==0)     return 0;
    LL n=x,m=sqrt(n),judge=1;
    for(LL i=3;i<=m;i+=2)
    {
	  if(n%i==0)
      {
	    judge=0;
        break;
      }
    }
    return judge;
}
```

### 埃式素数筛：

```c++
#include <bits/stdc++.h>
#define bbn 100005
using namespace std;
vector<int>prime;        //其中存储2-n的所有素数
bool is_prime[100005];   //存储第i项是否为素数
void find_prime(int n)
{
    int p=0;
    for(int i=2;i<=n;i++)
        is_prime[i]=true;
    is_prime[0]=is_prime[1]=false;
    for(int i=2;i<=n;i++)
    {
        if(is_prime[i])
        {
            prime.push_back(i);
            for(int j=2*i;j<=n;j+=i)
                is_prime[j]=false;
        }
    }
}
```

### 线性素数筛&&欧拉筛:

```c++
#include <bits/stdc++.h>
#define bbn 1000005
#define mod 1000000007 //1e9+7
typedef  long long int  LL;
using namespace std;
const int bbnn=1e8;
vector<int>prime;
bitset<bbnn>check;
void getprime(int n)
{
    int cnt=0;
    for(int i=2;i<=n;i++)
    {
        if(!check[i])
        {
            prime.push_back(i);
        }
        for(int j=0;j<prime.size();j++)
        {
            int ans=i*prime[j];
            if(ans>=n)
            {
                break;
            }
            check[ans]=1;
            if(i%prime[j]==0)
            {
                break;
            }
        }
    }
}
int main()
{
    std::ios::sync_with_stdio(0);
    int n,q;
    cin>>n>>q;
    getprime(n);
    for(int i=1; i<=q; i++)
    {
        int x;
        cin>>x;
        cout<<prime[x-1]<<endl;
    }
}

```



### 互质euler：

```c++
//输出比n小的且与n互质的正整数个数
LL euler(LL n)
{
	LL ans=n;
	for (LL i=2;i<=n;i++)
	{
		if(n%i==0)                     
        {
		 ans=ans/i*(i-1);//欧拉函数                 
        }
		while(n%i==0)//筛法选素数
		{
			n/=i;
		}
	}
	return ans;
}
```

线性素数筛&&欧拉筛：

```c++
const int bbn=1e8;
vector<int>prime;
bitset<bbn>check;
void getprime(int n)
{
    int cnt=0;
    for(int i=2;i<=n;i++)
    {
        if(!check[i])
        {
            prime.push_back(i);
        }
        for(int j=0;j<prime.size();j++)
        {
            int ans=i*prime[j];
            if(ans>=n)
            {
                break;
            }
            check[ans]=1;
            if(i%prime[j]==0)
            {
                break;
            }
        }
    }
}
int main()
{
    int n,q;
    cin>>n>>q;
    getprime(n);
    for(int i=1; i<=q; i++)
    {
        int x;
        cin>>x;
        cout<<prime[x-1]<<endl;
    }
}
```



### 快速幂：

```c++
LL quick_pow(LL a,LL b,LL mod)//a^b%mod
{
    LL ans=1;
    while(b>0)
    {
        if(b&1)
        {
            ans=ans*a%mod;
        }
        a=a*a%mod;
        b>>=1;
    }
    return ans;
}
```

### 大整数快速幂取模：

```c++
#include <bits/stdc++.h>
using namespace std;
const int mod=1e9+7;
long long quick_mod(long long a,long long b)
{
    long long ans=1;
    while(b){
        if(b&1){
            ans=(ans*a)%mod;
            b--;
        }
        b/=2;
        a=a*a%mod;
    }
    return ans;
}//内部也用快速幂
long long quickmod(long long a,char *b,int len)
{
    long long ans=1;
    while(len>0){
        if(b[len-1]!='0'){
            int s=b[len-1]-'0';
            ans=ans*quick_mod(a,s)%mod;
        }
        a=quick_mod(a,10)%mod;
        len--;
    }
    return ans;
}
int main(){
    char s[100050];
    int a;
    while(~scanf("%d",&a))         //求a^s%mod
    {
        scanf("%s",s);
        int len=strlen(s);
        printf("%I64d\n",quickmod(a,s,len));
    }
    return 0;
}
```



### 矩阵快速幂：

```c++
const int mod=1000000007;//1e9+7
LL t;
struct martix
{
   LL m[101][101]={};
};
void print(martix x,LL n)
{
    for(int i=1; i<=n; i++)
    {
        for(int j=1; j<=n; j++)
        {
            cout<<x.m[i][j]<<" ";
        }
        cout<<endl;
    }
}
martix mul(martix a,martix b,LL n)
{
    martix res;
    for(LL i=1; i<=n; i++)
    {
        for(LL j=1; j<=n; j++)
        {
            for(LL k=1; k<=n; k++)
            {
                res.m[i][j]+=a.m[i][k]*b.m[k][j];
                res.m[i][j]%=mod;
            }
        }
    }
    return res;
}
martix martix_quickpow(martix x,LL p,LL n)
{
    martix base=x;
    while(p)
    {
        if(p&1)
        {
            x=mul(x,base,n);
        }
        base=mul(base,base,n);
        p>>=1;
    }

    return x;
}
int main()
{
    LL n,k;
    scanf("%lld%lld",&n,&k);
    martix x;
    for(LL i=1; i<=n; i++)
    {
        for(LL j=1;j<=n;j++)
        {
             scanf("%lld",&x.m[i][j]);
        }
    }
    x=martix_quickpow(x,k-1,n);
    print(x,n);
}
```



### 扩展欧几里得：

```c++
#include <bits/stdc++.h>
#define bbn 100005
using namespace std;
int x,y;
int gcd(int a,int b)
{

	return b!=0?gcd(b,a%b):a;
}
int lcm(int a,int b)
{
	return a/gcd(a,b)*b;
}


int exgcd(int a, int b, int &x, int &y) {         //x，y初始为任意值，最后变为一组特解
    if(b == 0) {        //对应最终情况，a=gcd(a,b),b=0,此时x=1，y为任意数
        x = 1;
        y = 0;
        return a;
    }
    int r = exgcd(b, a % b, x, y);      //先递归到最终情况，再反推出初始情况
    int t = x; x = y; y = t - a / b * y;
    return r;     //gcd(a,b)
}
void RemainderEquation(int a,int b,int n)
{
    int X,Y,d;
    long long res;
    long long min_res;
    d=gcd(a,n);
    exgcd(a,n,X,Y);
    if(b%d == 0)
    {
        X = X * (b / d) % n;//得到同于方程一解
        for(int i = 0 ; i < d; i++)
        {
            res = (X + (i * (b/d))) % n;
            printf("%lld\n",res);             //输出所以解
        }
        min_res=(X%(n/d)+(n/d))%(n/d);
        cout<<min_res<<endl;       //输出最小解
    }else
    {
        printf("No Sulutions!\n");
    }
}

int main()
{
    RemainderEquation(3,5,4);


}

```

### 组合数计算：

```c++
LL c[100][100];
LL cnf(LL n,LL m)
{
    if(c[n][m]==0)
    {
        if(m==0||n==1||n==m)
        {
            c[n][m]=1;
            return c[n][m];
        }
        else
        {
            c[n-1][m-1]=cnf(n-1,m-1);
            c[n-1][m]=cnf(n-1,m);
            c[n][m]=cnf(n-1,m-1)+cnf(n-1,m);
            return c[n][m];
        }
    }
    else
    {
        return c[n][m];
    }
}
```

### 裴蜀定理:

```c++
#define bbn 200005
LL gcd(LL a,LL b)
{
    return b==0?a:gcd(b,a%b);
}
int main()
{
    LL n,a[bbn]= {};
    scanf("%lld",&n);
    for(LL i=1; i<=n; i++)
    {
        scanf("%lld",&a[i]);
        if(a[i]<0)
        {
            a[i]=-a[i];
        }
    }
    LL x=a[1];
    for(LL i=2; i<=n; i++)
    {
        x=gcd(x,a[i]);
    }
    printf("%lld\n",x);
}

```



## 图论：

### 二分图判定（1）:

```c++
#include<bits/stdc++.h>
#define bbn 105
using namespace std;
int g[bbn][bbn];
int used[bbn];
int n;
bool check()
{
    memset(used,-1,sizeof(used));
    queue<int>Q;
    Q.push(1);
    used[1]=0;
    while(!Q.empty())
    {
        int now=Q.front();
        for(int i=1; i<=n; i++)  //遍历所有点
        {
            if(g[now][i]==0)    //邻接矩阵存图
                continue;
            int v=i;
            if(used[v]==-1)
            {
                used[v]=(used[now]+1)%2;
                Q.push(v);
            }
            else
            {
                if(used[v]==used[now])
                    return false;
            }
        }
        Q.pop();
    }
    
    return true;
}
int main()
{

}

```



### 二分图匹配（匈牙利算法):

```c++
#include<bits/stdc++.h>
#define bbn 100005
using namespace std;
int n, m,eid;
int head[bbn];
int ans[bbn];
bool vis[bbn];
struct edge
{
    int v;
    int next;
} e[bbn];
void addedge(int u,int v)
{
    eid++;
    e[eid].v = v;
    e[eid].next = head[u];
    head[u] = eid;
}
void init()
{
    memset(head, -1, sizeof(head));
    eid = 0;
    int e;
    scanf("%d%d%d",&n,&m,&e);
    for(int i=1,u,v; i<=e; i++)
    {
        scanf("%d%d",&u,&v);
        addedge(u,v);
    }
}
bool dfs(int u)
{
    for (int i = head[u]; i!=-1; i=e[i].next)
    {
        int v = e[i].v;
        if (!vis[v])
        {
            vis[v] = 1;
            if (ans[v] == -1 || dfs(ans[v]))
            {
                ans[v] = u;
                return 1;
            }
        }
    }
    return 0;
}
int maxmatch()
{
    int cnt = 0;
    memset(ans, -1, sizeof(ans));
    for (int i = 1; i <= n; i++)
    {
        memset(vis, 0, sizeof(vis));
        cnt += dfs(i);
    }
    return cnt;
}
int main()
{
    init();
    printf("%d\n",maxmatch());
}

```



### 带权二分图匹配（KM算法）（1）:

```c++
#include<bits/stdc++.h>
#define bbn 105
using namespace std;
const int MAXN = 305;
const int INF = 0x3f3f3f3f;
int love[MAXN][MAXN];   // 记录每个妹子和每个男生的好感度
int ex_girl[MAXN];      // 每个妹子的期望值
int ex_boy[MAXN];       // 每个男生的期望值
bool vis_girl[MAXN];    // 记录每一轮匹配匹配过的女生
bool vis_boy[MAXN];     // 记录每一轮匹配匹配过的男生
int match[MAXN];        // 记录每个男生匹配到的妹子 如果没有则为-1
int slack[MAXN];        // 记录每个汉子如果能被妹子倾心最少还需要多少期望值
int N;
bool dfs(int girl)
{
    vis_girl[girl] = true;

    for (int boy = 0; boy < N; ++boy)
    {

        if (vis_boy[boy]) continue; // 每一轮匹配 每个男生只尝试一次

        int gap = ex_girl[girl] + ex_boy[boy] - love[girl][boy];

        if (gap == 0)    // 如果符合要求
        {
            vis_boy[boy] = true;
            if (match[boy] == -1 || dfs( match[boy] ))      // 找到一个没有匹配的男生 或者该男生的妹子可以找到其他人
            {
                match[boy] = girl;
                return true;
            }
        }
        else
        {
            slack[boy] = min(slack[boy], gap);  // slack 可以理解为该男生要得到女生的倾心 还需多少期望值 取最小值
        }
    }

    return false;
}
int KM()
{
    memset(match, -1, sizeof match);    // 初始每个男生都没有匹配的女生
    memset(ex_boy, 0, sizeof ex_boy);   // 初始每个男生的期望值为0

    // 每个女生的初始期望值是与她相连的男生最大的好感度
    for (int i = 0; i < N; ++i)
    {
        ex_girl[i] = love[i][0];
        for (int j = 1; j < N; ++j)
        {
            ex_girl[i] = max(ex_girl[i], love[i][j]);
        }
    }

    // 尝试为每一个女生解决归宿问题
    for (int i = 0; i < N; ++i)
    {

        fill(slack, slack + N, INF);    // 因为要取最小值 初始化为无穷大

        while (1)
        {
            // 为每个女生解决归宿问题的方法是 ：如果找不到就降低期望值，直到找到为止

            // 记录每轮匹配中男生女生是否被尝试匹配过
            memset(vis_girl, false, sizeof vis_girl);
            memset(vis_boy, false, sizeof vis_boy);

            if (dfs(i)) break;  // 找到归宿 退出

            // 如果不能找到 就降低期望值
            // 最小可降低的期望值
            int d = INF;
            for (int j = 0; j < N; ++j)
                if (!vis_boy[j]) d = min(d, slack[j]);

            for (int j = 0; j < N; ++j)
            {
                // 所有访问过的女生降低期望值
                if (vis_girl[j]) ex_girl[j] -= d;

                // 所有访问过的男生增加期望值
                if (vis_boy[j]) ex_boy[j] += d;
                // 没有访问过的boy 因为girl们的期望值降低，距离得到女生倾心又进了一步！
                else slack[j] -= d;
            }
        }
    }
    // 匹配完成 求出所有配对的好感度的和
    int res = 0;
    for (int i = 0; i < N; ++i)
        res += love[ match[i] ][i];

    return res;
}
int main()
{
    while (~scanf("%d", &N))
    {

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                scanf("%d", &love[i][j]);

        printf("%d\n", KM());
    }
    return 0;
}

```



### Prim:

```c++
#include<bits/stdc++.h>
#define bbn 500005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef  long long int  LL;
using namespace std;
int n,m,cnt;
int t,ans,now=1;
int head[bbn],dis[bbn];
bool vis[bbn];
struct E
{
    int v,w,next;
} e[bbn];
void addedge(int u,int v,int w)
{

    cnt++;
    e[cnt].v=v;
    e[cnt].w=w;
    e[cnt].next=head[u];
    head[u]=cnt;
}
void prim()
{
    for(int i=1; i<=n; i++)
    {
        dis[i]=maxint;
    }
    for(int i=head[1]; i; i=e[i].next)
    {
        dis[e[i].v]=min(dis[e[i].v],e[i].w);
    }
    while(1)
    {
        if(t==n-1)
        {
            break;
        }
        if(vis[now])
        {
            ans=-1;
            break;
        }
        t++;
        int minn=maxint;
        vis[now]=1;
        for(int i=1; i<=n; i++)
        {
            if(!vis[i]&&minn>dis[i])
            {
                minn=dis[i];
                now=i;
            }
        }
        ans+=minn;
        for(int i=head[now]; i; i=e[i].next)
        {
            int v=e[i].v;
            if(dis[v]>e[i].w&&!vis[v])
            {
                dis[v]=e[i].w;
            }
        }
    }
    if(ans==-1)
    {
        printf("orz\n");
    }
    else
    {
        printf("%d\n",ans);
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1; i<=m; i++)
    {
        int u,v,w;
        scanf("%d%d%d",&u,&v,&w);
        addedge(u,v,w);
        addedge(v,u,w);
    }
    prim();
}

```



### Kruskal:

```c++
#include <bits/stdc++.h>
#define bbn 1000001
using namespace std;
int n,m,sum,cnt;
int f[bbn];
struct Edge
{
    int from;
    int to;
    int dis;
} edge[bbn];
bool cmp(Edge a,Edge b)
{
    return a.dis<b.dis;
}
int father(int x)
{
    return f[x]!=x?father(f[x]):x;
}
void unionn(int x,int y)
{
    f[father(y)]=father(x);
}
void  kruskal()
{
    for(int i=1; i<=n; i++)
    {
        f[i]=i;
    }
    sort(edge+1,edge+m+1,cmp);
    for(int i=1; i<=m; i++)
    {
        if(cnt==n-1) break;
        if(father(edge[i].from)!=father(edge[i].to))
        {
            unionn(edge[i].from,edge[i].to);
            sum+=edge[i].dis;
            cnt++;
        }
    }
    if(cnt<n-1)
    {
        printf("orz\n");
    }
    else
    {
        printf("%d\n",sum);
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1; i<=m; i++)
    {
        scanf("%d%d%d",&edge[i].from,&edge[i].to,&edge[i].dis);
    }
    kruskal();

}

```



### Dijkstra:

```c++
#include <bits/stdc++.h>
#define bbn 1000010
#define bbnn 2147483647
#define mod 1000000007 //1e9+7
using namespace std;
typedef long long LL;
LL n,m,s,cnt;
LL head[bbn];
LL dis[bbn];
priority_queue<pair<LL,LL>,vector<pair<LL,LL>>,greater<pair<LL,LL>>> q;
struct edge
{
    LL v,w,nxt;
} e[2*bbn];
void addedge(LL u,LL v,LL w)
{
    e[++cnt]=(edge) {v,w,head[u]},head[u]=cnt;
}
void dijkstra(LL s)
{
    for(int i=1;i<=n;i++)
    {
        dis[i]=bbnn;
    }

   //memset(dis,0x3f,sizeof(dis));
    dis[s]=0,q.push(make_pair(0,s));
    while (!q.empty())
    {
        LL u=q.top().second;
        LL d=q.top().first;
        q.pop();
        if (d!=dis[u]) continue;
        for (LL i=head[u]; i; i=e[i].nxt)
        {
            LL v=e[i].v;
            LL w=e[i].w;
            if (dis[u]+w<dis[v])
            {
                dis[v]=dis[u]+w,q.push(make_pair(dis[v],v));
            }
        }
    }
}
int main()
{
    cin>>n>>m>>s;
    for (LL i=1; i<=m; i++)
    {
        LL u,v,w;
        cin>>u>>v>>w;
        addedge(u,v,w);
    }
    dijkstra(s);
    for(LL i=1; i<=n; i++)
    {
        cout<<dis[i]<<' ';
    }
    cout<<endl;
}

```

### Bellman-Ford(1)

```c++
#include<bits/stdc++.h>
#define bbn 105
using namespace std;
const int INF=0x3f3f3f3f;
struct edge
{
    int from;
    int to;
    int cost;
};
edge es[bbn];  //边
int d[bbn],V,E;    //最短距离，顶点数，边数
void bellman_ford(int s)
{
    memset(d,INF,sizeof(d));
    d[s]=0;
    while(true)
    {
        bool update = false;
        for(int i=0; i<E; i++)
        {
            edge e = es[i];
            if(d[e.from] != INF && d[e.to]>d[e.from]+e.cost)
            {
                d[e.to]=d[e.from]+e.cost;
                update = true;
            }
        }
        if(!update) break;
    }
}
bool find_negative_loop()   //返回true表示存在负圈
{
    memset(d,0,sizeof(d));
    for(int i=0; i<V; i++)
    {
        for(int j=0; j<E; j++)
        {
            edge e=es[j];
            if(d[e.to]>d[e.from]+e.cost)
                d[e.to]=d[e.from]+e.cost;
            if(i==V-1)           //如果第n次仍然更新了，则存在负圈
                return true;
            }
    }
    return false;
}
int main()
{

}

```



### Spfa:

```c++
#include<bits/stdc++.h>
#define bbn 1000005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef  long long int  LL;
using namespace std;
int n,m,s,cnt;
int dis[bbn],head[bbn],k[bbn];
bool vis[bbn];
struct E
{
    int v;
    int w;
    int next;
} e[bbn];
void addedge(int u,int v,int w)
{
    cnt++;
    e[cnt].v=v;
    e[cnt].w=w;
    e[cnt].next=head[u];
    head[u]=cnt;
}
void spfa(int s)
{
    memset(vis, 0, sizeof(vis));
    memset(dis, 0x3f, sizeof(dis));
    dis[s] = 0;
    vis[s] = 1;
    queue<int> q;
    q.push(s);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        /*
        if(k[u]>n)
        {
             cout<<"存在负环!"<<endl;
        }
        */
        vis[u] = 0;
        for (int i = head[u]; i; i = e[i].next)
        {
            int v = e[i].v;
            if (dis[u] + e[i].w < dis[v])
            {
                dis[v] = dis[u] + e[i].w;
                if (!vis[v])
                {
                    q.push(v);
                    k[v]++;
                    vis[v] = true;
                }
            }
        }
    }
}
int main()
{
    cin>>n>>m>>s;
    for (LL i=1; i<=m; i++)
    {
        LL u,v,w;
        cin>>u>>v>>w;
        addedge(u,v,w);
    }
    spfa(s);
    for(LL i=1; i<=n; i++)
    {
        cout<<dis[i]<<' ';
    }
    cout<<endl;
}

```



### Floyd:

```c++
#include <bits/stdc++.h>
#define bbn 105
using namespace std;
const int inf = 0x3f3f3f3f;
int g[bbn][bbn];  // 算法中的 G 矩阵
// 初始化 g 矩阵
int n,m;
void init()
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                g[i][j] = 0;
            }
            else
            {
                g[i][j] = inf;
            }
        }
    }
}
// 插入一条带权有向边
void insert(int u, int v, int w)
{
    g[u][v] = w;
}
// 核心代码
void floyd()
{
    for (int k = 0; k < n; ++k)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (g[i][k] +g[k][j] < g[i][j])
                {
                    g[i][j] = g[i][k] + g[k][j];
                }
            }
        }
    }
}
int main()
{
    cin>>n>>m;
    init();
    for(int i=1; i<=m; i++)
    {
        int  x,y;
        double z;
        cin>>x>>y>>z;
        insert(x,y,z);
        insert(y,x,z);
    }


}

```



### 负环:

```c++
#include<bits/stdc++.h>
#define bbn 1000005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef  long long int  LL;
using namespace std;
int n,m,cnt;
int dis[bbn],head[bbn],k[bbn];
bool vis[bbn];
struct E
{
    int v;
    int w;
    int next;
} e[bbn];
void addedge(int u,int v,int w)
{
    cnt++;
    e[cnt].v=v;
    e[cnt].w=w;
    e[cnt].next=head[u];
    head[u]=cnt;
}
void spfa(int s)
{
    memset(vis, 0, sizeof(vis));
    memset(dis, 0x3f, sizeof(dis));
    memset(k,0,sizeof(k));
    dis[s] = 0;
    vis[s] = 1;
    queue<int> q;
    q.push(s);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();

        if(k[u]>n)
        {
            cout<<"YES"<<endl;
            return;
        }
        vis[u] = 0;
        for (int i = head[u]; i; i = e[i].next)
        {
            int v = e[i].v;
            if (dis[u] + e[i].w < dis[v])
            {
                dis[v] = dis[u] + e[i].w;
                if (!vis[v])
                {
                    q.push(v);
                    k[v]++;
                    vis[v] = 1;
                }
            }
        }
    }
    cout<<"NO"<<endl;
}

void init()
{
    cnt=0;
    memset(head,0,sizeof(head));
}

int main()
{
    int t;
    cin>>t;
    while(t--)
    {
        init();
        cin>>n>>m;
        for (int i=1; i<=m; i++)
        {
            int u,v,w;
            cin>>u>>v>>w;
            addedge(u,v,w);
            if(w>=0)
            {
                addedge(v,u,w);
            }
        }
        spfa(1);
    }
}

```

### 拓扑排序:

```c++
#include <bits/stdc++.h>
#define bbn 105
using namespace std;
struct edge
{
    int to,next;
} e[bbn];
int p[bbn],eid;
void init()
{
    eid=0;
    memset(p,-1,sizeof(p));
}
void insert(int u,int v)
{
    e[eid].to=v;
    e[eid].next=p[u];
    p[u]=eid++;
}
int indegree[bbn];
int n,m;
int topo()
{
    queue<int>Q;
    for(int i=1; i<=n; i++)
    {
        if(indegree[i]==0)
            Q.push(i);
    }
    while(!Q.empty())
    {
        int now=Q.front();
        cout<<now<<' ';
        Q.pop();
        for(int i=p[now]; i!=-1; i=e[i].next)
        {
            int v=e[i].to;
            indegree[v]--;
            if(indegree[v]==0)
                Q.push(v);
        }
    }
    cout<<endl;
}
int main()
{
    init();
    cin>>n>>m;
    for(int i=1;i<=m;i++)
    {
        int x,y;
        cin>>x>>y;
        insert(x,y);
        indegree[y]++;
    }
    topo();
}

```

### Targan:

https://www.luogu.com.cn/problem/P3387

【模板】缩点

题目描述
给定一个 n个点 m 条边有向图，每个点有一个权值，求一条路径，使路径经过的点权值之和最大。你只需要求出这个权值和。
允许多次经过一条边或者一个点，但是，重复经过的点，权值只计算一次。
输出格式
共一行，最大的点权之和。

```c++
#include<bits/stdc++.h>
#define bbn 100005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000007 //1e6+7
typedef  long long int  LL;
using namespace std;

vector<int> g[10001];
int in,pd[bbn],dfn[bbn],low[bbn];
int tot,color[bbn],sum[bbn],f[bbn];
int st[bbn],top;
int n,m,w[bbn],x[bbn],y[bbn],ans;

void tarjan(int x)
{
    st[++top]=x;
    pd[x]=1;
    dfn[x]=low[x]=++in;
    for(int i=0; i<g[x].size(); i++)
    {
        int v=g[x][i];
        if(!dfn[v])
        {
            tarjan(v);
            low[x]=min(low[x],low[v]);
        }
        else if(pd[v])
        {
            low[x]=min(low[x],dfn[v]);
        }
    }
    if(dfn[x]==low[x])
    {
        tot++;
        while(st[top+1]!=x)
        {
            color[st[top]]=tot;
            sum[tot]+=w[st[top]];
            pd[st[top--]]=0;
        }
    }
}

void dfs(int x)
{
    if(f[x])
    {
        return;
    }
    f[x]=sum[x];
    int maxsum=0;
    for(int i=0; i<g[x].size(); i++)
    {
        if(!f[g[x][i]])
        {
            dfs(g[x][i]);
        }
        maxsum=max(maxsum,f[g[x][i]]);
    }
    f[x]+=maxsum;
}

void work()
{
    cin>>n>>m;
    for(int i=1; i<=n; i++)
    {
        cin>>w[i];
    }
    for(int i=1; i<=m; i++)
    {
        cin>>x[i]>>y[i];
        g[x[i]].push_back(y[i]);
    }
    for(int i=1; i<=n; i++)
    {
        if(!dfn[i])
        {
            tarjan(i);
        }
    }
    for(int i=1;i<=n;i++)
    {
        g[i].clear();
    }
    for(int i=1; i<=m; i++)
    {
        if(color[x[i]]!=color[y[i]])
        {
            g[color[x[i]]].push_back(color[y[i]]);
        }
    }
    for(int i=1; i<=tot; i++)
    {
        if(!f[i])
        {
            dfs(i);
            ans=max(ans,f[i]);
        }
    }
    cout<<ans<<'\n';
}

int main()
{
    work();
    return 0;
}

```



### Dinic:

```
#include<bits/stdc++.h>
#define bbn 100005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef long long int LL;
using namespace std;
int n,m,s,t,cnt=1;
LL max_flow;
int deep[bbn],head[bbn],cur[bbn];
struct edge
{
    int v,w,nxt;
} e[bbn];
void addedge(int u,int v,int w)
{
    e[++cnt]=(edge) {v,w,head[u]},head[u]=cnt;
}
int bfs()
{
    //cout<<"BFS"<<endl;
    for(int i=1; i<=n; i++)
    {
        deep[i]=0;
    }
    queue<int>q;
    q.push(s);
    deep[s]=1;
    while(q.size())
    {
        int now=q.front();
        q.pop();
        for(int i=head[now]; i; i=e[i].nxt)
        {
            int to=e[i].v,val=e[i].w;
            if(val&&(!deep[to]))
            {
                q.push(to);
                deep[to]=deep[now]+1;
                if(to==t)
                {
                    return 1;
                }
            }
        }
    }
    return 0;
}
int dfs(int u,int flow)
{
    // cout<<"DFS"<<endl;
    if(u==t)
    {
        return flow;
    }
    int rest=flow,k;
    for(int i=cur[u]; i&&rest; i=e[i].nxt)
    {
        cur[u]=i;
        int to=e[i].v,val=e[i].w;
        if(val&&(deep[to]==deep[u]+1))
        {
            int k=dfs(to,min(rest,val));
            if(!k)
            {
                deep[to]=0;
            }
            e[i].w-=k;
            e[i^1].w+=k;
            rest-=k;
        }
    }
    return flow-rest;
}
void solve()
{
    int flow=0;
    while(bfs())
    {
        for(int i=1; i<=n; i++)
        {
            cur[i]=head[i];
        }
        while(flow=dfs(s,1<<29))max_flow+=flow;
    }
    cout<<max_flow<<endl;
}
int main()
{
    scanf("%d%d%d%d",&n,&m,&s,&t);
    for(int i=1,u,v,w; i<=m; i++)
    {
        scanf("%d%d%d",&u,&v,&w);
        addedge(u,v,w);
        addedge(v,u,0);
    }
    solve();
}

```

### 最大团极大团Bron–Kerbosch（1）：

```c++
#include<bits/stdc++.h>
#define bbn 105
#define N 1010
/*
最大团 = 补图G的最大独立集数
———>最大独立集数 = 补图G'最大团
*/
//最大团模板
bool a[N][N];//a为图的邻接表(从1开始)
int ans, cnt[N], group[N], n, m, vis[N];//ans表示最大团，cnt[N]表示当前最大团的节点数，group[N]用以寻找一个最大团集合
bool dfs( int u, int pos )//u为当从前顶点开始深搜，pos为深搜深度（即当前深搜树所在第几层的位置）
{
    int i, j;
    for( i = u+1; i <= n; i++)//按递增顺序枚举顶点
    {
        if( cnt[i]+pos <= ans ) return 0;//剪枝
        if( a[u][i] )
        {
            // 与目前团中元素比较，取 Non-N(i)
            for( j = 0; j < pos; j++ ) if( !a[i][ vis[j] ] ) break;
            if( j == pos )
            {
                // 若为空，则皆与 i 相邻，则此时将i加入到 最大团中
                vis[pos] = i;//深搜层次也就是最大团的顶点数目，vis[pos] = i表示当前第pos小的最大团元素为i（因为是按增顺序枚举顶点 ）
                if( dfs( i, pos+1 ) ) return 1;
            }
        }
    }
    if( pos > ans )
    {
        for( i = 0; i < pos; i++ )
            group[i] = vis[i]; // 更新最大团元素
        ans = pos;
        return 1;
    }
    return 0;
}
void maxclique()//求最大团
{
    ans=-1;
    for(int i=n; i>0; i--)
    {
        vis[0]=i;
        dfs(i,1);
        cnt[i]=ans;
    }
}
int main()
{
    //freopen("D:\in.txt","r",stdin);
    int T;
    //scanf("%d",&T);
    while(~scanf("%d",&n))
    {
        if(n==0) break;
        //scanf("%d%d",&n,&m );
        int x, y;
        memset( a, 0, sizeof(a));
        /*for(int i = 0; i < m; i++)
        {
            scanf("%d%d",&x,&y);
            a[x][y] = a[y][x] = 1;
        }*/
        //相邻顶点间有边相连，模型转换成求 无向图 最大独立集。
        //要求原图的最大独立集，转化为求原图的补图的最大团(最大团顶点数量 = 补图的最大独立集)
        for(int i = 1; i <= n; i++)//求原图的补图
            for(int j = 1; j <= n; j++)
                scanf("%d",&a[i][j]);
        maxclique();//求最大团
        if( ans < 0 ) ans = 0;//ans表示最大团
        printf("%d\n", ans );
        /*for(int i = 0; i < ans; i++)
            printf( i == 0 ? "%d" : " %d", group[i]);//group[N]用以寻找一个最大团集合
        if( ans > 0 ) puts("");*/
    }
}


```



## 字符串：

### KMP:

```c++
#include<bits/stdc++.h>
#define bbn 1000005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef  long long int  LL;
using namespace std;
int kmp[bbn];
int len1,len2;
char a[bbn],b[bbn];
int main()
{
    cin>>a+1;
    cin>>b+1;
    len1=strlen(a+1);
    len2=strlen(b+1);
    for (int i=2,j=0; i<=len2; i++)
    {
        while(j>0&&b[i]!=b[j+1])
        {
            j=kmp[j];
        }
        if(b[j+1]==b[i])
        {
            j++;
            kmp[i]=j;
        }
    }
    for(int i=1,j=0; i<=len1; i++)
    {
        while(j>0&&b[j+1]!=a[i])
        {
            j=kmp[j];
        }
        if (b[j+1]==a[i])
        {
            j++;
        }
        if (j==len2)
        {
            printf("%d\n",i-len2+1);
            j=kmp[j];
        }
    }

    for (int i=1; i<=len2; i++)
    {
        printf("%d ",kmp[i]);
    }
}

```

### manacher:

```c++
#include <bits/stdc++.h>
#define bbn 1000001
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
const double eps=1e-7;
typedef long long int  LL;
using namespace std;

string x;
vector<char>s(22000002);
vector<int>p(22000002);
int manacher()//string x
{
    int ans=0,cnt=-1;
    s[++cnt]='!';
    s[++cnt]='|';
    for(int i=0; i<x.size(); i++)
    {
        s[++cnt]=x[i];
        s[++cnt]='|';
    }
    for(int t=1,r=0,mid=0; t<=cnt; t++)
    {
        if(t<=r)
        {
            p[t]=min(p[mid*2-t],r-t+1);
        }
        while(s[t-p[t]]==s[t+p[t]])
        {
            p[t]++;
        }
        if(t+p[t]>r)
        {
            r=t+p[t]-1;
            mid=t;
        }
        if(p[t]>ans)
        {
            ans=p[t];
        }
    }
    return ans-1;
}
main()
{
    cin>>x;
    cout<<manacher()<<endl;
}

```



## 动态规划：



###  01背包:

```c++
#include <bits/stdc++.h>
#define bbn 1000010
using namespace std;
int v,n,dp[bbn];
int w,c;
int main()
{
 cin>>v>>n;
 for(int i=1;i<=n;i++)
 {
     cin>>w>>c;
      if(w<=v)
     {
         for(int j=v;j>=w;j--)
         {
             dp[j]=max(dp[j],dp[j-w]+c);
         }
     }
 }
cout<<dp[v]<<endl;
}

```



### 完全背包:

```c++
#include <bits/stdc++.h>
#define bbn 10000010
using namespace std;
long long int v,n,dp[bbn],w,c;
int main()
{
 cin>>v>>n;
 for(int i=1;i<=n;i++)
 {
     cin>>w>>c;
      if(w<=v)
     {
         for(int j=w;j<=v;j++)
         {
             dp[j]=max(dp[j],dp[j-w]+c);
         }
     }
 }
cout<<dp[v]<<endl;
}

```



### 多重背包:

```c++
#include <bits/stdc++.h>
using namespace std;
int n,v,s,k,w0,c0,cnt,rest,w[100010],c[100010],dp[100010];
int main()
{
    cin>>n>>v;
    while(n--)
    {
        cin>>w0>>c0>>s;//重量为w0，价值为c0的物品有s件
        rest=s;k=1;//rest是剩余的物品数，k是从小到大的2的幂次
        while(rest>=k)//rest<k时结束，结束时按2的幂次恰好分完则rest=0，否则0<rest<k，要继续处理剩余的rest
        {
            w[++cnt]=w0*k;
            c[cnt]=c0*k;
            rest=rest-k;
            k=k*2;
        }
        if(rest>0){w[++cnt]=w0*rest;c[cnt]=c0*rest;}
    }
    for(int i=1;i<=cnt;i++)
        for(int j=v;j>=w[i];j--)
            dp[j]=max(dp[j],dp[j-w[i]]+c[i]);
    printf("%d\n",dp[v]);
    return 0;
}
```



### 混合背包:

```
#include <bits/stdc++.h>
using namespace std;
int n,v,s,k,w0,c0,cnt,rest,w[100010],c[100010],dp[100010];
int main()
{
    cin>>v>>n;
    while(n--)
    {
        cin>>w0>>c0>>s;//重量为w0，价值为c0的物品有s件
        if(s==0)//按完全背包处理
        {
            for(int j=w0;j<=v;j++)
            dp[j]=max(dp[j],dp[j-w0]+c0);
        }
        else//把多重背包转化为01背包，把s分解为2的幂次和，保存到w数组和c数组中，之和再按01背包处理
        {
            rest=s;k=1;//rest是剩余的物品数，k是从小到大的2的幂次
            while(rest>=k)//rest<k时结束，结束时按2的幂次恰好分完则rest=0，否则0<rest<k，要继续处理剩余的rest
            {
                w[++cnt]=w0*k;
                c[cnt]=c0*k;
                rest=rest-k;
                k=k*2;
            }
            if(rest>0){w[++cnt]=w0*rest;c[cnt]=c0*rest;}
        }
    }
    for(int i=1;i<=cnt;i++)//按01背包处理
        for(int j=v;j>=w[i];j--)
            dp[j]=max(dp[j],dp[j-w[i]]+c[i]);
    printf("%d\n",dp[v]);
    return 0;
}
```



### LCS:

```c++
P1439 【模板】最长公共子序列
5                       
3 2 1 4 5
1 2 3 4 5
最长公共子序列的长度 3

#include <bits/stdc++.h>
#define bbn 100001
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
const double eps=1e-7;
typedef  long long int  LL;
using namespace std;
int *up(int a[],int n)
{
   static int r[bbn] {};
    int dp[bbn]= {};
    dp[1]=a[1];
    int len=1;
    r[1]=1;
    for(int i=2; i<=n; i++)
    {
        if(dp[len]<a[i])
        {
            dp[++len]=a[i];
        }
        else
        {
            dp[lower_bound(dp+1,dp+len+1,a[i])-dp]=a[i];
        }
        r[i]=len;
    }
    return r;
}
int main()
{
    int n,a[bbn]= {},b[bbn]= {};
    cin>>n;
    for(int i=1,x; i<=n; i++)
    {
        cin>>x;
        a[x]=i;
    }
    for(int i=1; i<=n; i++)
    {
        cin>>b[i];
        b[i]=a[b[i]];
    }
    int *answer=up(b,n);
    cout<<answer[n]<<endl;
}

```



### LIS:

```c++
P1091 [NOIP2004 提高组] 合唱队形

#include <bits/stdc++.h>
#define bbn 100001
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
const double eps=1e-7;
typedef  long long int  LL;
using namespace std;
int *up(int a[],int n)
{
    static int r[bbn] {};
    int dp[bbn]= {};
    dp[1]=a[1];
    int len=1;
    r[1]=1;
    for(int i=2; i<=n; i++)
    {
        if(dp[len]<a[i])
        {
            dp[++len]=a[i];
        }
        else
        {
            dp[lower_bound(dp+1,dp+len+1,a[i])-dp]=a[i];
        }
        r[i]=len;
    }
    return r;
}
int *down(int a[],int n)
{
    static int d[bbn] {};
    int dp[bbn]= {};
    dp[1]=a[n];
    int len=1;
    d[1]=1;
    for(int i=n-1; i>=1; i--)
    {
        if(dp[len]<a[i])
        {
            dp[++len]=a[i];
        }
        else
        {
            dp[lower_bound(dp+1,dp+len+1,a[i])-dp]=a[i];
        }
        d[i]=len;
    }
    return d;
}
int main()
{
    int n,a[bbn]= {};
    cin>>n;
    for(int i=1; i<=n; i++)
    {
        cin>>a[i];

    }
    int *answer1=up(a,n);
    int *answer2=down(a,n);
    int minn=maxint;
    for(int i=1;i<=n;i++)
    {
        minn=min(minn,n-answer1[i]-answer2[i+1]);
    }
    cout<<minn<<endl;
}

```



## 数据结构：

### 并查集:

```c++
#include<bits/stdc++.h>
#define bbn 100005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef long long int LL;
using namespace std;
int n,m,a[bbn];

int fa(int x)
{
    return a[x]=a[x]==x?x:fa(a[x]);//路径压缩
}

void join(int x,int y)
{
    a[fa(y)]=fa(x);
}
int main()
{
    cin>>n>>m;
    for(int i=1; i<=n; i++)
    {
        a[i]=i;
    }
    for(int i=1,judge,x,y; i<=m; i++)
    {
        cin>>judge>>x>>y;
        if(judge==1)
        {
            join(x,y);
        }
        else
        {
            if(fa(x)==fa(y))
            {
                cout<<"Y"<<endl;
            }
            else
            {
                cout<<"N"<<endl;
            }
        }
    }
}

```



### ST表维护区间极差:

```c++
#include<bits/stdc++.h>
#define bbn 2000001
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef long long int LL;
using namespace std;
int a[bbn][22];
int b[bbn][22];
int st(int x,int y)
{
    int k=log2(y-x+1);
    return max(a[x][k],a[y-(1<<k)+1][k])-min(b[x][k],b[y-(1<<k)+1][k]);
}
int main()
{
    int n,q;
    scanf("%d%d",&n,&q);
    for(int i=1,x; i<=n; i++)
    {
        scanf("%d",&x);
        a[i][0]=x;
        b[i][0]=x;
    }
    for(int j=1; j<=21; j++)
    {
        for(int i=1; i+(1<<j)-1<=n; i++)
        {
            a[i][j]=max(a[i][j-1],a[i+(1<<(j-1))][j-1]);
            b[i][j]=min(b[i][j-1],b[i+(1<<(j-1))][j-1]);
        }
    }
    for(int i=1,x,y; i<=q; i++)
    {
        scanf("%d%d",&x,&y);
        cout<<st(x,y)<<endl;
    }
}

```

### 初级线段树:

```c++
//2021.8.24
//add(1,x,y,k); 在[x,y]区间(点)上加k  
//search(1,x,y);返回[x,y]区间(点)之和  
// 注意bbn大小(会RE||ME)

#include<bits/stdc++.h>
#define bbn 500001
using namespace std;
typedef long long int LL;
LL a[bbn];
struct T
{
    LL l,r;
    LL s,g;

} t[4*bbn];
void build(LL i,LL l,LL r)
{
    t[i].l=l;
    t[i].r=r;
    if(l==r)
    {
        t[i].s=a[l];
        return ;
    }
    LL mid=(l+r)>>1;
    build(i<<1,l,mid);
    build((i<<1)+1,mid+1,r);
    t[i].s=t[i<<1].s+t[(i<<1)+1].s;
}
void pushdown(LL i)
{
    if(t[i].g!=0)
    {
        t[i<<1].g+=t[i].g;
        t[i<<1].s+=t[i].g*(t[i<<1].r-t[i<<1].l+1);


        t[(i<<1)+1].g+=t[i].g;
        t[(i<<1)+1].s+=t[i].g*(t[(i<<1)+1].r-t[(i<<1)+1].l+1);

        t[i].g=0;
    }
}
LL search(LL i,LL l,LL r)
{
    if(l<=t[i].l&&t[i].r<=r)
    {
        return t[i].s;
    }
    if(t[i].r<l||t[i].l>r)
    {
        return 0;
    }
    pushdown(i);
    LL sum=0;
    LL mid=(t[i].l+t[i].r)>>1;
    if(l<=mid)
    {
        sum+=search(i<<1,l,r);
    }
    if(r>=mid+1)
    {
        sum+=search((i<<1)+1,l,r);
    }
    return sum;
}
void add(LL i,LL l,LL r,LL k)
{
    if(l<=t[i].l&&t[i].r<=r)
    {
        t[i].s+=k*(t[i].r-t[i].l+1);
        t[i].g+=k;
        return ;
    }
    pushdown(i);
    LL mid=(t[i].l+t[i].r)>>1;
    if(l<=mid)
    {
        add(i<<1,l,r,k);
    }
    if(r>=mid+1)
    {
        add((i<<1)+1,l,r,k);
    }

    t[i].s=t[i<<1].s+t[(i<<1)+1].s;
}
int main()
{
    LL n,m;
    scanf("%lld%lld",&n,&m);
    for(LL i=1; i<=n; i++)
    {
        scanf("%lld",&a[i]);
    }
    build(1,1,n);
    for(LL i=1; i<=m; i++)
    {
         LL x,y,z;
        scanf("%lld%lld%lld",&x,&y,&z);
        if(x==1)
        {
            add(1,y,y,z);
        }
        else if(x==2)
        {
            printf("%lld\n",search(1,y,z));
        }
    }
}

```

### 中级线段树:

```c++
#include<bits/stdc++.h>
#define bbn 500001
using namespace std;
typedef long long int LL;
LL a[bbn];
LL n,m,p;
struct T
{
    LL l,r;
    LL s,g,gg;

} t[4*bbn];
void build(LL i,LL l,LL r)
{
    t[i].l=l;
    t[i].r=r;
    t[i].g=0;
    t[i].gg=1;
    if(l==r)
    {
        t[i].s=a[l]%p;
        return ;
    }
    LL mid=(l+r)>>1;
    build(i*2,l,mid);
    build(i*2+1,mid+1,r);
    t[i].s=(t[i*2].s+t[i*2+1].s)%p;
}
void pushdown(LL i)
{
    t[i*2].s=(t[i*2].s*t[i].gg+t[i].g*(t[i*2].r-t[i*2].l+1))%p;
    t[i*2+1].s=(t[i*2+1].s*t[i].gg+t[i].g*(t[i*2+1].r-t[i*2+1].l+1))%p;
    //
    t[i*2].gg=(t[i*2].gg*t[i].gg)%p;
    t[i*2+1].gg=(t[i*2+1].gg*t[i].gg)%p;
    //
    t[i*2].g=(t[i*2].g*t[i].gg+t[i].g)%p;
    t[i*2+1].g=(t[i*2+1].g*t[i].gg+t[i].g)%p;
    t[i].g=0;
    t[i].gg=1;
}
LL search(LL i,LL l,LL r)
{
    if(t[i].r<l||t[i].l>r)
    {
        return 0;
    }
    if(l<=t[i].l&&t[i].r<=r)
    {
        return t[i].s;
    }
    pushdown(i);
    return (search(i*2,l,r)+search(i*2+1,l,r))%p;
}
void mul(LL i,LL l,LL r,LL k)
{
    if(l<=t[i].l&&t[i].r<=r)
    {
        t[i].s=(t[i].s*k)%p;
        t[i].g=(t[i].g*k)%p;
        t[i].gg=(t[i].gg*k)%p;
        return ;
    }
    pushdown(i);
    LL mid=(t[i].l+t[i].r)>>1;
    if(l<=mid)
    {
        mul(i*2,l,r,k);
    }
    if(r>=mid+1)
    {
        mul(i*2+1,l,r,k);
    }
    t[i].s=(t[i*2].s+t[i*2+1].s)%p;
}

void add(LL i,LL l,LL r,LL k)
{
    if(l<=t[i].l&&t[i].r<=r)
    {
        t[i].g=(t[i].g+k)%p;
        t[i].s=(t[i].s+k*(t[i].r-t[i].l+1))%p;
        return ;
    }
    pushdown(i);
    LL mid=(t[i].l+t[i].r)>>1;
    if(l<=mid)
    {
        add(i*2,l,r,k);
    }
    if(r>=mid+1)
    {
        add(i*2+1,l,r,k);
    }

    t[i].s=(t[i*2].s+t[i*2+1].s)%p;
}
int main()
{

    scanf("%lld%lld%lld",&n,&m,&p);
    for(LL i=1; i<=n; i++)
    {
        scanf("%lld",&a[i]);
    }
    build(1,1,n);
    for(LL i=1; i<=m; i++)
    {
        LL judge;
        scanf("%lld",&judge);
        if(judge==1)
        {
            LL x,y,z;
            scanf("%lld%lld%lld",&x,&y,&z);
            mul(1,x,y,z);
        }
        else if(judge==2)
        {
            LL x,y,z;
            scanf("%lld%lld%lld",&x,&y,&z);
            add(1,x,y,z);
        }
        else if(judge==3)
        {
            LL x,y;
            scanf("%lld%lld",&x,&y);
            printf("%lld\n",search(1,x,y));
        }
    }
}

```

### 树状数组:

```c++
#include<bits/stdc++.h>
#define bbn 500001
typedef long long int LL;
using namespace std;
int n,m,tree[2000010];
int lowbit(int k)
{
    return k & -k;
}
void add(int x,int k)
{
    while(x<=n)
    {
        tree[x]+=k;
        x+=lowbit(x);
    }
}
int sum(int x)
{
    int ans=0;
    while(x!=0)
    {
        ans+=tree[x];
        x-=lowbit(x);
    }
    return ans;
}
int main()
{
    cin>>n>>m;
    for(int i=1; i<=n; i++)
    {
        int a;
        scanf("%d",&a);
        add(i,a);
    }
    for(int i=1; i<=m; i++)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        if(a==1)
            add(b,c);
        if(a==2)
            cout<<sum(c)-sum(b-1)<<endl;
    }
}

```

### 滑动窗口维护最大值最小值:

```c++
#include <iostream>
#include <deque>
#define bbn 1000005
using namespace std;

int n,k;
struct E
{
    int id,val;
} a[bbn];

void f_min()
{
    deque<E>q;
    for(int i=1; i<=n; i++)
    {
        while(!q.empty()&&a[i].val<=q.back().val)
        {
            q.pop_back();
        }
        q.push_back(a[i]);
        while(q.front().id<=i-k)
        {
            q.pop_front();
        }
        if(i>=k)
        {
             printf("%d ",q.front().val);
        }
    }
    printf("\n");

}
void f_max()
{
    deque<E>q;
    for(int i=1; i<=n; i++)
    {
        while(!q.empty()&&a[i].val>=q.back().val)
        {
            q.pop_back();
        }
        q.push_back(a[i]);
        while(q.front().id<=i-k)
        {
            q.pop_front();
        }
        if(i>=k)
        {
             printf("%d ",q.front().val);
        }
    }
     printf("\n");
}

void work()
{
    scanf("%d%d",&n,&k);
    for(int i=1; i<=n; i++)
    {
        scanf("%d",&a[i].val);
        a[i].id=i;
    }
    f_min();
    f_max();
}

int main()
{
    work();
    return 0;
}

```

### 单调栈:

```c++
#include<bits/stdc++.h>
#define bbn 3000005
#define maxint 2147483647
#define maxLLint 9223372036854775807
#define mod 1000000007 //1e9+7
typedef long long int LL;
using namespace std;
int n;
int a[bbn];
int q[bbn];
stack<int>s;
int main()
{
    cin>>n;
    for(int i=1; i<=n; i++)
    {
        cin>>a[i];
    }
    for(int i=n; i>=1; i--)
    {
        while(!s.empty()&&a[s.top()]<=a[i])
        {
            s.pop();
        }
        q[i]=s.empty()?0:s.top();
        s.push(i);
    }
    for(int i=1;i<=n;i++)
    {
        cout<<q[i]<<' ';
    }
    cout<<endl;
}

```

### 

## 计算几何：



### 旋转卡壳：

```c++
include <bits/stdc++.h>
using namespace std;
const int N=1000+10;
typedef long long ll;
struct Point
{
    ll x, y;
    Point() {}
    Point(ll _x, ll _y): x(_x), y(_y) {}
    void in()
    {
        cin>>x>>y;
    }
    bool operator<(Point b)const
    {
        return x < b.x || ( x == b.x && y < b.y );
    }
    Point operator−(Point b)
    {
        return Point(x − b.x, y − b.y);
    }
    ll operator^(Point b)
    {
        return x * b.y − y * b.x;
    }
} p[N];
struct Line
{
    int x, y;
    double k;
    bool operator<(Line b)const
    {
        return k < b.k;
    }
};

ll ans = 0x3f3f3f3f3f3f3f3f;
int n,rk[N],id[N];
vector<Line>l;
int main()
{
    cin>>n;
    for( int i = 1 ; i <= n ; i++ )p[i].in();
    sort( p + 1, p + n + 1 );
    for( int i = 1 ; i <= n ; i++ ) id[ i ] = rk[ i ] = i;
    for( int i = 1 ; i < n ; i++ )//k属 于[−pi/2,pi/2)
        for( int j = i + 1 ; j <= n ; j++ )
            l.push_back({i,j,atan2(p[j].y−p[i].y,p[j].x−p[i].x)});
    sort(l.begin(),l.end());
    for(auto i:l)
    {
        int a=i.x,b=i.y;
        if( id[ a ] > id[ b ] ) swap( a, b );
        if(id[a]!=1)ans=min(ans, abs( (p[b]−p[a])^(p[b]−p[rk[id[a]−1]]) ));
        swap(id[a],id[b]);
        swap(rk[ id[a] ],rk[ id[b] ]);
    }
    cout<<ans/2<<(ans%2?”.50”:”.00”);
    return 0;
}

```

