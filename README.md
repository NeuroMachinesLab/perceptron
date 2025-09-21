# Pure Java Rumelhart's Multilayer Perceptron

![multilayer perceptron](https://github.com/user-attachments/assets/faa01fd2-0c30-4386-983e-9eaef7eb707f)

Usage [example](examples/src/main/java/ai/neuromachines/examples/TrainingSample.java).

Implements pure java [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
with one of the [Activation Functions](https://en.wikipedia.org/wiki/Activation_function)
and [Backpropagation training algorithm](https://en.wikipedia.org/wiki/Backpropagation).

## Activation Functions

This [activation function](https://en.wikipedia.org/wiki/Activation_function) are implemented currently:
- Identity
- Sigmoid
- ReLU
- PReLU
- ELU
- SiLU
- Softplus
- Tanh

## Backpropagation Algorithm

Perceptron is trained by Backpropagation algorithm. Key formulas are described below.

At each iteration weight between i-th and j-th node is changed by:
```
Δw_ij = - η * y_i * δ_j   (1)
```
where: <br>
`η` - [learning rate](https://en.wikipedia.org/wiki/Learning_rate); <br>
`y_i` - output of i-th node; <br>
`δ_j` - delta coefficient for j-th node.

Delta coefficient for j-th node is calculation depends on location of the node. For output layer node it evaluated by:
```
δ_j = (e_j - y_j) * f'(S_j)  (2)
```
where: <br>
`e_j` - expected output for j-th node; <br>
`y_j` - real output for j-th node; <br>
`f'(S_j)` - derivative of activation function; <br>
`S_j` - input signal for j-th node.

Node input signal is:
```
S_j = sum(y_i * w_ij)  (3)
```
where: <br>
`y_i` - output of i-th node (located closer to the input layer), which connected to j-th node (located closer to output layer); <br>
`w_ij` - weight between i-th and j-th node.

For hidden layer node Delta coefficient evaluated by:
```
δ_j = f'(S_j) * sum(w_jk * δ_k)  (4)
```
where: <br>
`w_jk` - weight between j-th node (located closer to input layer) and k-th node (located closer to output layer); <br>
`δ_k` - delta coefficient for j-th node.
