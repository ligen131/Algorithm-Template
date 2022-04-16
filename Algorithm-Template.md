# Algorithm Template

[TOC]

![https://zhuanlan.zhihu.com/p/454647571](img/Algorithm-tree.jpg)

## 1 编译命令 Compile Command

Sublime Text Build File for C++14 in Windows

```JSON
{
"encoding": "utf-8",
"working_dir": "$file_path",
"shell_cmd": "g++ -Wall -Wextra -std=c++14  -Wl,--stack=80000000 \"$file_name\" -o \"$file_base_name\"",
"file_regex": "^(..[^:]*):([0-9]+):?([0-9]+)?:? (.*)$",
"selector": "source.c++", "variants": 
[
    {   
    "name": "Compile",
		"shell_cmd": "g++ -Wall -Wextra -std=c++14 -Wl,--stack=80000000 \"$file_name\" -o \"$file_base_name\" -DLOCAL_LIGEN",
    },
    {   
    "name": "Run",
        "shell_cmd": "start cmd /c \"\"${file_path}/${file_base_name}\" & pause\""
    },
    {   
    "name": "Compile_and_Run",
        "shell_cmd": "g++ -Wall -Wextra -std=c++14 -Wl,--stack=80000000 \"$file_name\" -o \"$file_base_name\" -DLOCAL_LIGEN && start cmd /c \"\"${file_path}/${file_base_name}\" & pause\""
    },
]
}
```

## 2 头文件 Head Files

```cpp
// copyright (c) 2022 ligen131 <1353055672@qq.com>
// #pragma GCC optimize(3)
#include <iostream>
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <bitset>
#include <vector>
#include <assert.h>
#include <ctime>
using namespace std;

#define ll long long
#define ull unsigned long long
#define infll (long long)(1e18)
#define infint (int)(1 << 30)
#define mod (int)(1e9 + 7)
#define FOR(a, b, c) for(int a = b; a <= c; ++a)
#define FORD(a, b, c) for(int a = b; a >= c; --a)
#define YES(v) ((v) ? puts("Yes") : puts("No"))

template <typename T>
inline T Max(const T &a, const T &b) {return a > b ? a : b;}
template <typename T>
inline T Min(const T &a, const T &b) {return a < b ? a : b;}
template <typename T>
inline T Abs(const T &a) {return a > 0 ? a : - a;}
template <typename T>
inline void Swap(T &a, T &b) {T c = a; a = b; b = c;}
template <typename T>
inline T MOD(T &a) {return a %= mod;}
template <typename T>
inline T MOD(const T &a) {return a % mod;}

// #define READ_BUF_LEN 1<<20
// char READ_BUF[READ_BUF_LEN], *READ_P1 = READ_BUF, *READ_P2 = READ_BUF;
// #define getchar() (READ_P1 == READ_P2 && (READ_P2 = (READ_P1 = READ_BUF) + fread(READ_BUF, 1, READ_BUF_LEN, stdin), READ_P1 == READ_P2) ? EOF : *READ_P1++)
template <typename T>
inline T read(T &a) {
	char c; bool neg = false; a = 0;
	for(c = getchar(); c < '0' || c > '9'; neg |= c == '-', c = getchar());
	for(; c >= '0' && c <= '9'; a = a * 10 - '0' + c, c = getchar());
	if (neg) a = -a;
	return a;
}
template <typename T, typename... Args>
inline void read(T &a, Args&... args) {read(a); read(args...);}
inline long long read() {long long a; return read(a);}
char WRITE_BUF[40];
template <typename T>
inline void write(T a) {
	if (!a) return putchar('0'), void();
	if (a < 0) putchar('-'), a = -a;
	int len = 0;
	while (a) WRITE_BUF[++len] = a % 10 + '0', a /= 10;
	for(int i = len; i; --i) putchar(WRITE_BUF[i]);
}
inline void write_() {return;}
template <typename T, typename... Args>
inline void write_(T a, Args... args) {write(a); putchar(' '); write_(args...);}
inline void writeln() {putchar('\n'); return;}
template <typename T, typename... Args>
inline void writeln(T a, Args... args) {write(a); if (sizeof...(args)) putchar(' '); writeln(args...);}

template <typename T>
T gcd(const T &a, const T &b) {return a == 0 ? b : gcd(b % a, a);}
inline long long lcm(const long long &a, const long long &b) {return 1ll * a / gcd(a, b) * b;}
inline long long Pow(long long a, long long n) {
	long long ans = 1;
	while(n) {
		if (n & 1) ans = 1ll * ans * a % mod;
		a = 1ll * a * a % mod;
		n >>= 1;
	}
	return ans;
}
//-------------------------------Head Files-------------------------------//
#define mn 200020

signed main() {
#ifdef LOCAL_LIGEN
	freopen("D:\\Code\\0.in","r",stdin);
	//freopen("0.out","w",stdout);
	const double PROGRAM_BEGIN_TIME = clock();
#endif
	int a,b;
	read(a, b);
	Swap(a, b);
	writeln(a, b, a + b, Max(a, b), Min(a, b), Abs(a));
#ifdef LOCAL_LIGEN
	printf("Time: %.0lfms\n", clock() - PROGRAM_BEGIN_TIME);
	fclose(stdin); fclose(stdout);
#endif
}
```

## 3 基础 Basic

### 3.1 排序

#### 3.1.1 选择排序

#### 3.1.2 冒泡排序

#### 3.1.3 插入排序

#### 3.1.4 归并排序

#### 3.1.5 桶排序

#### 3.1.6 基数排序

#### 3.1.7 *希尔排序

### 3.2 高精度

### 3.3 搜索

#### 3.3.1 折半搜索 Meet in the Middle

#### 3.3.2 *DLX

#### 3.3.3 *A*

## 4 数据结构 Data Structure

### 4.1 单调队列

### 4.2 单调栈

### 4.3 树

#### 4.3.1 树的重心

#### 4.3.2 树的直径

#### 4.3.3 最近公共祖先 LCA

### 4.4 线段树

#### 4.4.1 线段树

#### 4.4.2 扫描线

#### 4.4.3 可持久化线段树 / 主席树

#### 4.4.4 线段树合并

#### 4.4.5 *李超线段树

### 4.5 树状数组

#### 4.5.1 树状数组

#### 4.5.2 树状数组求第 k 大

#### 4.5.3 多维树状数组

### 4.6 平衡树

#### 4.6.1 Splay

#### 4.6.2 Treap

#### 4.6.3 Fhq Treap

#### 4.6.4 *可持久化平衡树

#### 4.6.5 *替罪羊树

### 4.7 *树套树

#### 4.7.1 *线段树套线段树

#### 4.7.2 *线段树套平衡树

### 4.8 树链剖分

#### 4.8.1 重链剖分

#### 4.8.2 *长链剖分

### 4.9 笛卡尔树

### 4.10 树的启发式合并 DSU on Tree

### 4.11 *动态树 Link Cut Tree

### 4.12 *虚树

### 4.13 *KD 树

### 4.14 *树的哈希

### 4.15 *析合树

### 4.16 并查集

#### 4.16.1 并查集

#### 4.16.2 带权并查集

#### 4.16.3 *可持久化并查集

### 4.17 分块

#### 4.17.1 链上分块

#### 4.17.2 莫队

#### 4.17.3 *树上分块

### 4.18 C++ STL

#### 4.18.1 pair

#### 4.18.2 vector

#### 4.18.3 priority_queue

#### 4.18.4 set / multiset

#### 4.18.5 map

#### 4.18.6 bitset

#### 4.18.7 *pbds

### 4.19 ST 表 / RMQ

## 5 图论 Graph Theory

### 5.1 最短路

#### 5.1.1 传递闭包 / Floyd

#### 5.1.2 Bellman-Ford

#### 5.1.3 Dijkstra

#### 5.1.4 *最短路径树

### 5.2 最小生成树

#### 5.2.1 Prim

#### 5.2.2 Kruskal

#### 5.2.3 Boruvka

### 5.3 点 / 边双连通分量

#### 5.3.1 Tarjan

#### 5.3.2 割点 / 割边

#### 5.3.3 *圆方树

#### 5.3.4 *Kosaraju

### 5.4 二分图

#### 5.4.1 判定

#### 5.4.2 匈牙利

#### 5.4.3 *KM

#### 5.4.4 *Hopcraft-karp

### 5.5 网络流

#### 5.5.1 最大流 / 最小割

####

## 6 动态规划 Dynamic Program

## 7 字符串 String

## 8 数学 Mathematics

## 9 几何 Geometry

## 10 其他 Others