package ai.neuromachines.network.layer;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.function.ActivationFunc;

/**
 * Intermediate and Output layers representation.
 * This layer in comparison with {@link SensorLayer} has input values, weights and activation function.
 */
public class ResponseLayer implements IntermediateLayer {
    private final Layer previous;
    // 1-st row contains all weights for 1-st node of current layer to the previous layer
    private final float[][] weight;
    // sum of input signals of all nodes in layer
    private float[] inputSum;
    private final ActivationFunc func;
    // current output values of all nodes in layer
    private float[] output;

    /**
     * Creates layer with specified connection weights.
     *
     * @param weights  weights for connections with nodes of the previous layer
     *                 (row count equals to current layer node count, cols count equals to previous layer node count)
     * @param previous previous layer
     * @param func     activation function
     */
    public static ResponseLayer of(float[][] weights, Layer previous, ActivationFunc func) {
        return new ResponseLayer(weights, previous, func);
    }

    /**
     * Creates layer with random connection weights.
     *
     * @param nodeCnt  current layer node count
     * @param previous previous layer
     * @param func     activation function
     */
    public static ResponseLayer of(int nodeCnt, Layer previous, ActivationFunc func) {
        float[][] weights = randomWeights(nodeCnt, previous.nodeCount());
        return new ResponseLayer(weights, previous, func);
    }

    private static float[][] randomWeights(int rows, int cols) {
        float[][] a = new float[rows][cols];
        for (float[] row : a) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (float) Math.random();
            }
        }
        return a;
    }

    private ResponseLayer(float[][] weights, Layer previous, ActivationFunc func) {
        Assert.isTrue(weights.length > 0, "At least one node expected");
        for (float[] row : weights) {
            Assert.isTrue(row.length == previous.nodeCount(), "Incorrect weights count");
        }
        this.previous = previous;
        this.weight = weights;
        this.func = func;
        int nodeCnt = weights.length;
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
        this.inputSum = Matrix.multiply(weight, previous.output());
        return output = Matrix.applyFunc(inputSum, func.function());
    }

    @Override
    public float[][] weights() {
        return weight;
    }
}
