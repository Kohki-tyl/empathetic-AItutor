# 機械翻訳レビュー用対訳

構造・保護対象の検査済み。意味の忠実性は未レビュー。

## math_train_1119 (train)

### 1: user

原文:

Mr. Madoff invests 1000 dollars in a fund that compounds annually at a constant interest rate.  After three years, his investment has grown to 1225 dollars.  What is the annual interest rate, as a percentage?  (Round your answer to the nearest integer.)

訳文:

マドフ氏は、毎年一定の利率で複利運用されるファンドに1000ドルを投資します。3年後、投資額は1225ドルになりました。年利は何パーセントですか。（答えは整数に四捨五入してください。）

### 2: assistant

原文:

Let $r$ be the annual interest rate.  Then after three years, Mr. Madoff's investment is $1000 \cdot \left( 1 + \frac{r}{100} \right)^3$, so \[1000 \cdot \left( 1 + \frac{r}{100} \right)^3 = 1225.\]Then \[\left( 1 + \frac{r}{100} \right)^3 = 1.225,\]so \[1 + \frac{r}{100} = \sqrt[3]{1.225} = 1.069987 \dots,\]which means $r = \boxed{7}$, to the nearest integer.

訳文:

$r$を年利とします。すると、3年後のマドフ氏の投資額は$1000 \cdot \left( 1 + \frac{r}{100} \right)^3$なので、\[1000 \cdot \left( 1 + \frac{r}{100} \right)^3 = 1225.\]したがって、\[\left( 1 + \frac{r}{100} \right)^3 = 1.225,\]となります。すると、\[1 + \frac{r}{100} = \sqrt[3]{1.225} = 1.069987 \dots,\]なので、整数に四捨五入すると$r = \boxed{7}$です。

## math_train_1120 (train)

### 1: user

原文:

February $1$, $2008$ is a Friday.  The last Friday of February $2008$ will be February $x$, $2008$.  If $2008$ is a leap year, what is the value of $x$?

訳文:

$2008$年2月$1$日は金曜日です。$2008$年2月の最後の金曜日は、$2008$年2月$x$日になります。$2008$年がうるう年だとすると、$x$の値はいくつですか？

### 2: assistant

原文:

If 2008 is a leap year, that means there are 29 days in February, and February 29 is $28=7\cdot4$ days after February 1, so it is also a Friday. Thus, $x=\boxed{29}$.

訳文:

2008年がうるう年なら、2月は29日あり、2月29日は2月1日から$28=7\cdot4$日後なので、これも金曜日です。したがって、$x=\boxed{29}$。

## math_train_1278 (train)

### 1: user

原文:

What is the units digit when $5^4$ is multiplied out?

訳文:

$5^4$を展開して掛け合わせたとき、1の位は何ですか？

### 2: assistant

原文:

The units digit of any positive integer power of 5 is $\boxed{5}$.

(Note: this claim may be proved by induction, since the units digit of $5n$ is 5 whenever the units digit of $n$ is 5.)

訳文:

5の任意の正の整数乗の1の位は$\boxed{5}$です。

（注：これは帰納法で証明できます。$n$の1の位が5であるとき、$5n$の1の位は5だからです。）

## math_train_1337 (train)

### 1: user

原文:

A triangle with sides measuring 8, 15 and 17 units is inscribed in a circle. What is the radius of the circle, in units? Express your answer as a decimal to the nearest tenth.

訳文:

辺の長さが8、15、17単位の三角形が円に内接しています。円の半径は何単位ですか。答えは小数第１位までで表してください。

### 2: assistant

原文:

An $8-15-17$ triangle is always right. It follows that the hypotenuse is a diameter of the circle, and the radius must be half the hypotenuse, or $\frac{17}{2} = \boxed{8.5}$.

訳文:

$8-15-17$の三角形は必ず直角三角形です。したがって、斜辺は円の直径であり、半径は斜辺の半分、つまり$\frac{17}{2} = \boxed{8.5}$です。

## math_train_1351 (train)

### 1: user

原文:

Casey's shop class is making a golf trophy.  He has to paint $300$ dimples on a golf ball.  If it takes him $2$ seconds to paint one dimple, how many minutes will he need to do his job?

訳文:

Caseyの工作の授業では、ゴルフのトロフィーを作っています。ゴルフボールに$300$個のディンプルを塗らなければなりません。ディンプルを一つ塗るのに$2$秒かかるとすると、作業に何分必要でしょうか？

### 2: assistant

原文:

At $2$ seconds per dimple, it takes $300 \times 2 = 600$ seconds to paint them.  Since there are $60$ seconds in a minute, he will need $600 \div 60 = \boxed{10}$ minutes.

訳文:

ディンプル一つあたり$2$秒かかるので、すべて塗るには$300 \times 2 = 600$秒かかります。1分は$60$秒なので、必要なのは$600 \div 60 = \boxed{10}$分です。

## math_train_1616 (train)

### 1: user

原文:

An infinite geometric series has common ratio $\frac{-1}{3}$ and sum $25.$  What is the second term of the sequence?

訳文:

無限等比級数の公比は$\frac{-1}{3}$で、和は$25.$です。この数列の第2項は何ですか？

### 2: assistant

原文:

The second term seems difficult to calculate directly, so we will first find the value of the first term. Let the first term be $a$.  Because the sum of the series is $25,$ we have  \[25= \frac{a}{1-\left(\frac{-1}{3}\right)} = \frac{a}{\frac{4}{3}} = \frac{3a}{4}.\]Thus, $a=\frac{100}{3}.$ Now, we can calculate the second term knowing the value of the first. The second term $ar$ is \[ar=\left( \frac{100}{3} \right)\left(\frac{-1}{3}\right)=\boxed{\frac{-100}{9}} .\]

訳文:

第2項を直接求めるのは難しそうなので、まず初項を求めましょう。初項を$a$とします。級数の和は$25,$なので、\[25= \frac{a}{1-\left(\frac{-1}{3}\right)} = \frac{a}{\frac{4}{3}} = \frac{3a}{4}.\]したがって、$a=\frac{100}{3}.$です。これで初項の値がわかったので、第2項を計算できます。第2項$ar$は\[ar=\left( \frac{100}{3} \right)\left(\frac{-1}{3}\right)=\boxed{\frac{-100}{9}} .\]です。

## math_train_1486 (train)

### 1: user

原文:

What is the smallest integer $x$ for which $x<2x-7$ ?

訳文:

$x<2x-7$となるような最小の整数$x$は何ですか？

### 2: assistant

原文:

Adding $7$ and subtracting $x$ from both sides of the inequality, we have that $7<x$. The smallest value $x$ that satisfies this inequality is $x = \boxed{8}$.

訳文:

不等式の両辺に$7$を足し、$x$を引くと、$7<x$となります。この不等式を満たす$x$の最小値は$x = \boxed{8}$です。

## math_train_1162 (train)

### 1: user

原文:

In right $\triangle ABC$ with hypotenuse $\overline{AB}$, $AC = 12$, $BC = 35$, and $\overline{CD}$ is the altitude to $\overline{AB}$. Let $\omega$ be the circle having $\overline{CD}$ as a diameter. Let $I$ be a point outside $\triangle ABC$ such that $\overline{AI}$ and $\overline{BI}$ are both tangent to circle $\omega$. The ratio of the perimeter of $\triangle ABI$ to the length $AB$ can be expressed in the form $\frac {m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m + n$.

訳文:

斜辺が$\overline{AB}$である直角$\triangle ABC$において、$AC = 12$、$BC = 35$であり、$\overline{CD}$は$\overline{AB}$への高さである。$\overline{CD}$を直径とする円を$\omega$とする。$\triangle ABC$の外部にある点$I$で、$\overline{AI}$と$\overline{BI}$がどちらも円$\omega$に接するものとする。$\triangle ABI$の周の長さと長さ$AB$の比は、$m$と$n$が互いに素な正の整数であるとき、$\frac {m}{n}$の形で表せる。$m + n$を求めよ。

### 2: assistant

原文:

Let $O$ be center of the circle and $P$,$Q$ be the two points of tangent such that $P$ is on $BI$ and $Q$ is on $AI$. We know that $AD:CD = CD:BD = 12:35$.
Since the ratios between corresponding lengths of two similar diagrams are equal, we can let $AD = 144, CD = 420$ and $BD = 1225$. Hence $AQ = 144, BP = 1225, AB = 1369$ and the radius $r = OD = 210$.
Since we have $\tan OAB = \frac {35}{24}$ and $\tan OBA = \frac{6}{35}$ , we have $\sin {(OAB + OBA)} = \frac {1369}{\sqrt {(1801*1261)}},$$\cos {(OAB + OBA)} = \frac {630}{\sqrt {(1801*1261)}}$.
Hence $\sin I = \sin {(2OAB + 2OBA)} = \frac {2*1369*630}{1801*1261}$. let $IP = IQ = x$ , then we have Area$(IBC)$ = $(2x + 1225*2 + 144*2)*\frac {210}{2}$ = $(x + 144)(x + 1225)* \sin {\frac {I}{2}}$. Then we get $x + 1369 = \frac {3*1369*(x + 144)(x + 1225)}{1801*1261}$.
Now the equation looks very complex but we can take a guess here. Assume that $x$ is a rational number (If it's not then the answer to the problem would be irrational which can't be in the form of $\frac {m}{n}$) that can be expressed as $\frac {a}{b}$ such that $(a,b) = 1$. Look at both sides; we can know that $a$ has to be a multiple of $1369$ and not of $3$ and it's reasonable to think that $b$ is divisible by $3$ so that we can cancel out the $3$ on the right side of the equation.
Let's see if $x = \frac {1369}{3}$ fits. Since $\frac {1369}{3} + 1369 = \frac {4*1369}{3}$, and $\frac {3*1369*(x + 144)(x + 1225)}{1801*1261} = \frac {3*1369* \frac {1801}{3} * \frac {1261*4}{3}} {1801*1261} = \frac {4*1369}{3}$. Amazingly it fits!
Since we know that $3*1369*144*1225 - 1369*1801*1261 < 0$, the other solution of this equation is negative which can be ignored. Hence $x = 1369/3$.
Hence the perimeter is $1225*2 + 144*2 + \frac {1369}{3} *2 = 1369* \frac {8}{3}$, and $BC$ is $1369$. Hence $\frac {m}{n} = \frac {8}{3}$, $m + n = \boxed{11}$.

訳文:

円の中心を$O$とし、接点を$P$、$Q$とする。ただし、$P$は$BI$上にあり、$Q$は$AI$上にある。$AD:CD = CD:BD = 12:35$がわかっている。
対応する長さの比は相似な図形どうしで等しいので、$AD = 144, CD = 420$および$BD = 1225$とおける。したがって、$AQ = 144, BP = 1225, AB = 1369$であり、半径は$r = OD = 210$である。
$\tan OAB = \frac {35}{24}$および$\tan OBA = \frac{6}{35}$なので、$\sin {(OAB + OBA)} = \frac {1369}{\sqrt {(1801*1261)}},$$\cos {(OAB + OBA)} = \frac {630}{\sqrt {(1801*1261)}}$である。
したがって、$\sin I = \sin {(2OAB + 2OBA)} = \frac {2*1369*630}{1801*1261}$。$IP = IQ = x$とおくと、面積$(IBC)$ = $(2x + 1225*2 + 144*2)*\frac {210}{2}$ = $(x + 144)(x + 1225)* \sin {\frac {I}{2}}$となる。すると、$x + 1369 = \frac {3*1369*(x + 144)(x + 1225)}{1801*1261}$を得る。
この方程式はとても複雑に見えるが、ここで推測してみよう。$x$は有理数だと仮定する（そうでなければ、問題の答えは無理数となり、$\frac {m}{n}$の形にはならない）。$(a,b) = 1$となるような$\frac {a}{b}$と表せるとする。両辺を見ると、$a$は$1369$の倍数で、$3$の倍数ではないことがわかる。また、方程式の右辺にある$3$を約分できるよう、$b$は$3$で割り切れると考えるのが妥当だ。
$x = \frac {1369}{3}$が成り立つか確かめよう。$\frac {1369}{3} + 1369 = \frac {4*1369}{3}$であり、$\frac {3*1369*(x + 144)(x + 1225)}{1801*1261} = \frac {3*1369* \frac {1801}{3} * \frac {1261*4}{3}} {1801*1261} = \frac {4*1369}{3}$。驚いたことに、成り立つ！
$3*1369*144*1225 - 1369*1801*1261 < 0$がわかっているので、この方程式のもう一つの解は負であり、無視できる。したがって、$x = 1369/3$。
したがって周の長さは$1225*2 + 144*2 + \frac {1369}{3} *2 = 1369* \frac {8}{3}$であり、$BC$は$1369$である。したがって、$\frac {m}{n} = \frac {8}{3}$、$m + n = \boxed{11}$。

## math_train_1112 (train)

### 1: user

原文:

Two fair, six-sided dice are rolled.  What is the probability that the sum of the two numbers showing is between 3 and 11, inclusive?

訳文:

公平な六面体のサイコロを二つ振ります。出た目の合計が3から11までの範囲に含まれる確率はいくらですか？

### 2: assistant

原文:

We instead find the probability that the sum of the numbers showing is not between 3 and 11.  Since each die's face contains the numbers 1-6, we see that this can only be the case if we roll two 1s or two 6s.  Thus the chance that the sum is not between 3 and 11 is $\frac{1}{6} \cdot \frac{1}{6} + \frac{1}{6} \cdot \frac{1}{6}$, or $\frac{1}{18}$.  So our desired probability is $1-\frac{1}{18} = \boxed{\frac{17}{18}}$.

訳文:

代わりに、出た目の合計が3から11までの範囲に含まれない確率を求めます。それぞれのサイコロの面には1から6までの数字が書かれているので、これは二つとも1が出るか、二つとも6が出る場合に限られます。したがって、合計が3から11までの範囲に含まれない確率は$\frac{1}{6} \cdot \frac{1}{6} + \frac{1}{6} \cdot \frac{1}{6}$、つまり$\frac{1}{18}$です。よって、求める確率は$1-\frac{1}{18} = \boxed{\frac{17}{18}}$です。

## math_train_1546 (train)

### 1: user

原文:

If
\[\begin{pmatrix} 1 & 2 & a \\ 0 & 1 & 4 \\ 0 & 0 & 1 \end{pmatrix}^n = \begin{pmatrix} 1 & 18 & 2007 \\ 0 & 1 & 36 \\ 0 & 0 & 1 \end{pmatrix},\]then find $a + n.$

訳文:

もし
\[\begin{pmatrix} 1 & 2 & a \\ 0 & 1 & 4 \\ 0 & 0 & 1 \end{pmatrix}^n = \begin{pmatrix} 1 & 18 & 2007 \\ 0 & 1 & 36 \\ 0 & 0 & 1 \end{pmatrix},\]なら、$a + n.$を求めよ。

### 2: assistant

原文:

Let $\mathbf{A} = \begin{pmatrix} 1 & 2 & a \\ 0 & 1 & 4 \\ 0 & 0 & 1 \end{pmatrix}.$  Then we can write $\mathbf{A} = \mathbf{I} + \mathbf{B},$ where
\[\mathbf{B} = \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix}.\]Note that
\[\mathbf{B}^2 = \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 & 8 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}\]and
\[\mathbf{B}^3 = \mathbf{B} \mathbf{B}^2 = \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} \begin{pmatrix} 0 & 0 & 8 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} = \mathbf{0}.\]Then by the Binomial Theorem,
\begin{align*}
\mathbf{A}^n &= (\mathbf{I} + \mathbf{B})^n \\
&= \mathbf{I}^n + \binom{n}{1} \mathbf{I}^{n - 1} \mathbf{B} + \binom{n}{2} \mathbf{I}^{n - 2} \mathbf{B}^2 + \binom{n}{3} \mathbf{I}^{n - 3} \mathbf{B}^3 + \dots + \mathbf{B}^n \\
&= \mathbf{I} + n \mathbf{B} + \frac{n(n - 1)}{2} \mathbf{B}^2 \\
&= \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} + n \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} + \frac{n(n - 1)}{2} \begin{pmatrix} 0 & 0 & 8 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \\
&= \begin{pmatrix} 1 & 2n & an + 4n(n - 1) \\ 0 & 1 & 4n \\ 0 & 0 & 1 \end{pmatrix}.
\end{align*}Hence, $2n = 18,$ $an + 4n(n - 1) = 2007,$ and $4n = 36.$  Solving, we find $a = 191$ and $n = 9,$ so $a + n = \boxed{200}.$

Note: We can expand $(\mathbf{I} + \mathbf{B})^{2016}$ using the Binomial Theorem because the matrices $\mathbf{B}$ and $\mathbf{I}$ commute, i.e. $\mathbf{B} \mathbf{I} = \mathbf{I} \mathbf{B}.$  In general, expanding a power of $\mathbf{A} + \mathbf{B}$ is difficult.  For example,
\[(\mathbf{A} + \mathbf{B})^2 = \mathbf{A}^2 + \mathbf{A} \mathbf{B} + \mathbf{B} \mathbf{A} + \mathbf{B}^2,\]and without knowing more about $\mathbf{A}$ and $\mathbf{B},$ this cannot be simplified.

訳文:

$\mathbf{A} = \begin{pmatrix} 1 & 2 & a \\ 0 & 1 & 4 \\ 0 & 0 & 1 \end{pmatrix}.$とおく。すると、$\mathbf{A} = \mathbf{I} + \mathbf{B},$と書ける。ここで
\[\mathbf{B} = \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix}.\]注意すると、
\[\mathbf{B}^2 = \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 & 8 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}\]および
\[\mathbf{B}^3 = \mathbf{B} \mathbf{B}^2 = \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} \begin{pmatrix} 0 & 0 & 8 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} = \mathbf{0}.\]である。すると二項定理より、
\begin{align*}
\mathbf{A}^n &= (\mathbf{I} + \mathbf{B})^n \\
&= \mathbf{I}^n + \binom{n}{1} \mathbf{I}^{n - 1} \mathbf{B} + \binom{n}{2} \mathbf{I}^{n - 2} \mathbf{B}^2 + \binom{n}{3} \mathbf{I}^{n - 3} \mathbf{B}^3 + \dots + \mathbf{B}^n \\
&= \mathbf{I} + n \mathbf{B} + \frac{n(n - 1)}{2} \mathbf{B}^2 \\
&= \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} + n \begin{pmatrix} 0 & 2 & a \\ 0 & 0 & 4 \\ 0 & 0 & 0 \end{pmatrix} + \frac{n(n - 1)}{2} \begin{pmatrix} 0 & 0 & 8 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \\
&= \begin{pmatrix} 1 & 2n & an + 4n(n - 1) \\ 0 & 1 & 4n \\ 0 & 0 & 1 \end{pmatrix}.
\end{align*}したがって、$2n = 18,$、$an + 4n(n - 1) = 2007,$および$4n = 36.$である。これを解くと、$a = 191$、$n = 9,$が得られるので、$a + n = \boxed{200}.$

注：行列$\mathbf{B}$と$\mathbf{I}$は可換、すなわち$\mathbf{B} \mathbf{I} = \mathbf{I} \mathbf{B}.$であるため、二項定理を使って$(\mathbf{I} + \mathbf{B})^{2016}$を展開できる。一般に、$\mathbf{A} + \mathbf{B}$のべき乗を展開するのは難しい。たとえば、
\[(\mathbf{A} + \mathbf{B})^2 = \mathbf{A}^2 + \mathbf{A} \mathbf{B} + \mathbf{B} \mathbf{A} + \mathbf{B}^2,\]であり、$\mathbf{A}$と$\mathbf{B},$についてさらに分かっていなければ、これを簡単にすることはできない。

## math_train_1260 (train)

### 1: user

原文:

Four concentric circles are drawn with radii of 1, 3, 5 and 7. The inner circle is painted black, the ring around it is white, the next ring is black and the outer ring is white. What is the ratio of the black area to the white area? Express your answer as a common fraction.

訳文:

半径が1、3、5、7の同心円が四つ描かれています。内側の円は黒く塗られ、その周りの環は白く、次の環は黒く、外側の環は白く塗られています。黒い部分の面積と白い部分の面積の比はいくつですか。答えは既約分数で表してください。

### 2: assistant

原文:

The areas of the four circles are $\pi, 9\pi, 25\pi$ and $49\pi$. The areas of the two black regions are $\pi$ and $25\pi - 9\pi = 16\pi$, for a total black area of $\pi + 16\pi = 17\pi$. The areas of the two white regions are $9\pi - \pi = 8\pi$ and $49\pi - 25\pi = 24\pi$, for a total white area of $8\pi + 24\pi = 32\pi$. The ratio of the black area to the white area is $17\pi/32\pi = \boxed{\frac{17}{32}}.$

訳文:

四つの円の面積は$\pi, 9\pi, 25\pi$と$49\pi$です。黒い部分二つの面積は$\pi$と$25\pi - 9\pi = 16\pi$で、黒い部分の合計面積は$\pi + 16\pi = 17\pi$です。白い部分二つの面積は$9\pi - \pi = 8\pi$と$49\pi - 25\pi = 24\pi$で、白い部分の合計面積は$8\pi + 24\pi = 32\pi$です。黒い部分の面積と白い部分の面積の比は$17\pi/32\pi = \boxed{\frac{17}{32}}.$

## math_train_1239 (train)

### 1: user

原文:

The graphs of $y=x^4$ and $y=5x^2-6$ intersect at four points with $x$-coordinates $\pm \sqrt{m}$ and $\pm \sqrt{n}$, where $m > n$. What is $m-n$?

訳文:

$y=x^4$と$y=5x^2-6$のグラフは、$m > n$であるとき、$x$座標が$\pm \sqrt{m}$および$\pm \sqrt{n}$となる四つの点で交わります。$m-n$はいくつですか？

### 2: assistant

原文:

At the intersection points, the $y$-coordinates of the two graphs must be equal, so we have the equation $x^4=y=5x^2-6$, or $x^4=5x^2-6$. Putting all the terms on one side, we get $x^4-5x^2+6=0$. Factoring, we get $(x^2-3)(x^2-2)=0$, so $x^2-3=0 \Rightarrow x=\pm \sqrt{3}$ or $x^2-2=0 \Rightarrow x=\pm \sqrt{2}$. Thus, $m=3$ and $n=2$ and $m-n=\boxed{1}$.

訳文:

交点では、二つのグラフの$y$座標が等しくなければならないので、方程式$x^4=y=5x^2-6$、すなわち$x^4=5x^2-6$が得られます。すべての項を片側に移すと、$x^4-5x^2+6=0$となります。因数分解すると、$(x^2-3)(x^2-2)=0$となるので、$x^2-3=0 \Rightarrow x=\pm \sqrt{3}$または$x^2-2=0 \Rightarrow x=\pm \sqrt{2}$です。したがって、$m=3$、$n=2$であり、$m-n=\boxed{1}$です。

## math_train_1226 (train)

### 1: user

原文:

Compute: $15 + 8^2 - 3$.

訳文:

計算してください：$15 + 8^2 - 3$。

### 2: assistant

原文:

Recalling that exponents precede addition in the order of operations, we simplify $15+8^2-3=15+64-3=\boxed{76}$.

訳文:

指数は演算の順序で足し算より先に計算することを思い出すと、$15+8^2-3=15+64-3=\boxed{76}$ と簡単にできます。

## math_train_1178 (train)

### 1: user

原文:

A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?

訳文:

公園の縮尺図では、一インチが800フィートを表しています。図上で4.75インチの長さがある線分は、何フィートを表しますか？

### 2: assistant

原文:

Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\times800=\frac{19}{4}\cdot800=19\cdot200=\boxed{3800}$ feet.

訳文:

4.75インチの線分では、一インチごとに800フィートを表すので、線分全体では$4.75\times800=\frac{19}{4}\cdot800=19\cdot200=\boxed{3800}$フィートを表します。

## math_train_1544 (train)

### 1: user

原文:

Every card in a deck has a picture of one shape - circle, square, or triangle, which is painted in one of the three colors - red, blue, or green. Furthermore, each color is applied in one of three shades - light, medium, or dark. The deck has 27 cards, with every shape-color-shade combination represented. A set of three cards from the deck is called complementary if all of the following statements are true:
i. Either each of the three cards has a different shape or all three of the card have the same shape.
ii. Either each of the three cards has a different color or all three of the cards have the same color.
iii. Either each of the three cards has a different shade or all three of the cards have the same shade.
How many different complementary three-card sets are there?

訳文:

デッキの各カードには、円、正方形、三角形のいずれか一つの図形が描かれており、赤、青、緑のいずれか一色で塗られています。さらに、それぞれの色には、明るい、中間、暗いのいずれか一つの濃さがあります。デッキには27枚のカードがあり、図形・色・濃さのあらゆる組み合わせがそろっています。デッキから選んだ三枚のカードの組は、次のすべての条件を満たすとき、補完的と呼ばれます。
i. 三枚のカードの図形がすべて異なるか、三枚すべてが同じ図形である。
ii. 三枚のカードの色がすべて異なるか、三枚すべてが同じ色である。
iii. 三枚のカードの濃さがすべて異なるか、三枚すべてが同じ濃さである。
補完的な三枚のカードの組は何通りありますか？

### 2: assistant

原文:

Case 1: All three attributes are the same. This is impossible since sets contain distinct cards.
Case 2: Two of the three attributes are the same. There are ${3\choose 2}$ ways to pick the two attributes in question. Then there are $3$ ways to pick the value of the first attribute, $3$ ways to pick the value of the second attribute, and $1$ way to arrange the positions of the third attribute, giving us ${3\choose 2} \cdot 3 \cdot 3 = 27$ ways.
Case 3: One of the three attributes are the same. There are ${3\choose 1}$ ways to pick the one attribute in question, and then $3$ ways to pick the value of that attribute. Then there are $3!$ ways to arrange the positions of the next two attributes, giving us ${3\choose 1} \cdot 3 \cdot 3! = 54$ ways.
Case 4: None of the three attributes are the same. We fix the order of the first attribute, and then there are $3!$ ways to pick the ordering of the second attribute and $3!$ ways to pick the ordering of the third attribute. This gives us $(3!)^2 = 36$ ways.
Adding the cases up, we get $27 + 54 + 36 = \boxed{117}$.

訳文:

場合1：三つの属性がすべて同じ。組には異なるカードが含まれるため、これは不可能です。
場合2：三つの属性のうち二つが同じ。該当する二つの属性を選ぶ方法は${3\choose 2}$通りです。次に、最初の属性の値を選ぶ方法は$3$通り、二番目の属性の値を選ぶ方法は$3$通り、三番目の属性の位置を並べる方法は$1$通りなので、${3\choose 2} \cdot 3 \cdot 3 = 27$通りです。
場合3：三つの属性のうち一つが同じ。該当する属性を選ぶ方法は${3\choose 1}$通りで、その属性の値を選ぶ方法は$3$通りです。次に、残り二つの属性の位置を並べる方法は$3!$通りなので、${3\choose 1} \cdot 3 \cdot 3! = 54$通りです。
場合4：三つの属性に同じものはありません。最初の属性の順序を固定すると、二番目の属性の順序を選ぶ方法は$3!$通り、三番目の属性の順序を選ぶ方法は$3!$通りです。したがって、$(3!)^2 = 36$通りです。
これらの場合を合計すると、$27 + 54 + 36 = \boxed{117}$となります。

## math_train_1157 (train)

### 1: user

原文:

How many ways are there to put 5 balls in 3 boxes if the balls are not distinguishable but the boxes are?

訳文:

区別できない5個のボールを、箱は区別できる3個の箱に入れる方法は何通りありますか？

### 2: assistant

原文:

Since the balls are indistinguishable, we must only count the number of balls in the different boxes.

There are $3$ ways to arrange the balls as $(5,0,0)$ (specifically, box 1 can have 5, box 2 can have 5, box 3 can have 5).

There are $3! = 6$ to arrange $(4,1,0)$ and $3! = 6$ ways to arrange $(3,2,0)$; in each case, we must choose one of the 3 boxes to have the largest number of balls, and also one of the remaining two boxes to be left empty.

However, there are only $3$ ways to arrange $(3,1,1)$, and $3$ ways to arrange $(2,2,1)$; in each case, we must choose one box to have the `different' number of balls (3 in the $(3,1,1)$ case and 1 in the $(2,2,1)$ case).

This gives a total of $3 + 6 + 6 + 3 + 3 = \boxed{21}$ arrangements.

訳文:

ボールは区別できないので、それぞれの箱に入れるボールの個数だけを数えればよいです。

ボールを$(5,0,0)$のように並べる方法は$3$通りあります（具体的には、箱1には5個、箱2には5個、箱3には5個入れられます）。

$(4,1,0)$を並べる方法は$3! = 6$通り、$(3,2,0)$を並べる方法は$3! = 6$通りあります。どちらの場合も、最大個数のボールを入れる箱を3個の箱から一つ選び、残りの二つの箱のうち一つを空にする必要があります。

ただし、$(3,1,1)$を並べる方法は$3$通りしかなく、$(2,2,1)$を並べる方法も$3$通りです。どちらの場合も、「異なる」個数のボールを入れる箱を一つ選ぶ必要があります（$(3,1,1)$の場合は3個、$(2,2,1)$の場合は1個です）。

これで、並べ方の合計は$3 + 6 + 6 + 3 + 3 = \boxed{21}$通りになります。
