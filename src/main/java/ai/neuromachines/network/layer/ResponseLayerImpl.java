package ai.neuromachines.network.layer;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.function.ActivationFunc;

class ResponseLayerImpl implements ResponseLayer {
    private final Layer previous;
    // 1-st row contains all weights for 1-st node of current layer to the previous layer
    private final float[][] weight;
    // sum of input signals of all nodes in layer
    private float[] inputSum;
    private final ActivationFunc func;
    // current output values of all nodes in layer
    private float[] output;

    ResponseLayerImpl(float[][] weights, Layer previous, ActivationFunc func) {
        Assert.isTrue(weights.length > 0, "At least one node expected");
        for (float[] row : weights) {
            Assert.isTrue(row.length == previous.nodeCount(), "Incorrect weights count");
        }
        this.previous = previous;
        this.weight = weights;
        int nodeCnt = weights.length;
        this.inputSum = new float[nodeCnt];
        this.func = func;
        this.output = new float[nodeCnt];
    }

    @Override
    public ActivationFunc activationFunc() {
        return func;
    }

    @Override
    public int nodeCount() {
        return output.length;
    }

    @Override
    public float[] input() {
        return inputSum;
    }

    @Override
    public float[] output() {
        return output;
    }

    @Override
    public void propagate() {
        updateInput();
        output = Matrix.applyFunc(inputSum, func.function());
    }

    void updateInput() {
        inputSum = Matrix.multiply(weight, previous.output());
    }

    @Override
    public float[][] weights() {
        return weight;
    }
}
