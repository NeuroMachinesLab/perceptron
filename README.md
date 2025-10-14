# Pure Java Rumelhart's Multilayer Perceptron

![multilayer perceptron](https://github.com/user-attachments/assets/1872114f-4727-43fe-a0ca-9c661a66c071)

No more fat JAR required for starting from scratch. JAR's total size is about 50 kB!

Implements pure java [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
with one of the [Activation Functions](https://en.wikipedia.org/wiki/Activation_function)
and [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) training algorithm.

📰 See an [example](examples/src/main/java/ai/neuromachines/examples/TrainingSample.java) of use.

## Activation Functions

This activation functions ([[1]](https://en.wikipedia.org/wiki/Activation_function) and
[[2]](https://ru.wikipedia.org/wiki/%D0%A4%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F_%D0%B0%D0%BA%D1%82%D0%B8%D0%B2%D0%B0%D1%86%D0%B8%D0%B8))
are implemented currently:
- ArcTan
- Bent Identity
- ELU
- Gaussian
- Heaviside
- Identity
- ISRLU
- ISRU
- Leaky ReLU
- ReLU
- Sigmoid
- SiLU
- Sinc
- Sin
- Softmax
- Softplus
- Softsign
- Tanh

## Backpropagation Algorithm

Perceptron is trained by Backpropagation algorithm. Key formulas are described below.

At each iteration weight between i-th and j-th node is changed by:
```
Δw_ij = - η * y_i * δ_j                (1)
```
where: <br>
`η` - [learning rate](https://en.wikipedia.org/wiki/Learning_rate); <br>
`y_i` - output of i-th node; <br>
`δ_j` - delta coefficient for j-th node.

Delta coefficient for j-th node is calculation depends on location of the node
and type of [loss function](https://en.wikipedia.org/wiki/Loss_function).

For output layer node with [Softmax](https://en.wikipedia.org/wiki/Activation_function) activation function
and [Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) loss function
delta is [evaluated](https://habr.com/ru/articles/155235) by:
```
δ_j = y_j - e_j                        (2)
```
where: <br>
`y_j` - real output for j-th node; <br>
`e_j` - expected output for j-th node.

For output layer node with other type of activation function, except Softmax, and
[Least Squares](https://en.wikipedia.org/wiki/Least_squares) loss function
delta is [evaluated](https://en.wikipedia.org/wiki/Backpropagation) by:
```
δ_j = (y_j - e_j) * f'(S_j)            (3)
```
where: <br>
`f'(S_j)` - derivative of activation function; <br>
`S_j` - input signal for j-th node.

Node input signal is:
```
S_j = sum(y_i * w_ij)                  (4)
```
where: <br>
`y_i` - output of i-th node (located closer to the input layer), which connected to j-th node (located closer to output layer); <br>
`w_ij` - weight between i-th and j-th node.

For hidden layer node Delta coefficient evaluated by:
```
δ_j = f'(S_j) * sum(w_jk * δ_k)        (5)
```
where: <br>
`w_jk` - weight between j-th node (located closer to input layer) and k-th node (located closer to output layer); <br>
`δ_k` - delta coefficient for k-th node (located closer to output layer);
        the coefficient `δ_k` has already been calculated earlier by according to formula (3) or (4)
        if k-th node in output layer, and by formula (5) in previous iteration if k-th node in other layers.  

## How to Use Jar Library

There are 2 cases for get JAR package (~40 Kb).

1. Get JAR from GitHub [Packages](https://github.com/orgs/NeuroMachinesLab/packages?repo_name=perceptron).
In this case you [should](https://stackoverflow.com/questions/72732582/using-github-packages-without-personal-access-token)
use GitHub Personal Access Token (PAT). First, add repository to you maven project
```xml
<repositories>
  <repository>
    <id>central</id>
    <url>https://repo.maven.apache.org/maven2</url>
    <snapshots>
      <enabled>false</enabled>
    </snapshots>
  </repository>
  <repository>
    <id>github</id>
    <url>https://maven.pkg.github.com/NeuroMachinesLab/perceptron</url>
  </repository>
</repositories>
```
Secondly, configure GitHub Personal Access Token (PAT) by this
[Tutorial](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-apache-maven-registry).
After that you can add package to dependencies
```xml
<dependency>
    <groupId>ai.neuromachines</groupId>
    <artifactId>perceptron</artifactId>
    <version>3.0</version>
</dependency>
```

2. Or you can get JAR package from [JitPack](https://jitpack.io/#NeuroMachinesLab/perceptron) repository without PAT.
Add repository to you maven project
```xml
<repositories>
    <repository>
        <id>central</id>
        <name>Central Repository</name>
        <url>https://repo.maven.apache.org/maven2</url>
        <snapshots>
            <enabled>false</enabled>
        </snapshots>
    </repository>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>
```
and add dependency
```xml
<dependency>
    <groupId>com.github.NeuroMachinesLab</groupId>
    <artifactId>perceptron</artifactId>
    <version>3.0</version>
</dependency>
```
Please note that the groupId is different from the package on GitHub Packages.
This is expected, this is how JitPack works.
