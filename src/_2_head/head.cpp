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