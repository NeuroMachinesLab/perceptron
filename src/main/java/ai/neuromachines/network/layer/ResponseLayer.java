package ai.neuromachines.network.layer;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.Constants;
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
    private final Backpropagation backpropagation;

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
        this.backpropagation = new Backpropagation(nodeCnt);
    }

    public int nodeCount() {
        return output.length;
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

    @Override
    public void correctWeights(float[] expectedOutput) {
        Assert.isTrue(expectedOutput.length == nodeCount(), "Incorrect excepted output count");
        if (inputSum == null) {
            throw new IllegalStateException("Call output() first, there is no calculated output signal for weighs correcting");
        }
        float[] delta = backpropagation.calculateLastLayerDelta(expectedOutput);
        updateWeights(delta);
        updatePreviousLayerWeights();
    }

    private void correctIntermediateLayerWeights(ResponseLayer nextLayer) {
        float[] delta = backpropagation.calculateIntermediateLayerDelta(nextLayer);
        updateWeights(delta);
        updatePreviousLayerWeights();
    }

    private void updateWeights(float[] delta) {
        assert delta.length == nodeCount() : "Incorrect delta count";
        assert weight.length == nodeCount() : "Incorrect weight rows count";

        float[] prevOut = previous.output();
        for (int j = 0, cnt = nodeCount(); j < cnt; j++) {  // current layer
            assert weight[j].length == prevOut.length : "Incorrect previous layer node count";
            for (int i = 0; i < prevOut.length; i++) {  // previous layer
                float weightDelta = -Constants.TRAINING_VELOCITY * prevOut[i] * delta[j];
                weight[j][i] += weightDelta;
            }
        }
    }

    private void updatePreviousLayerWeights() {
        if (previous instanceof ResponseLayer rl) {
            rl.correctIntermediateLayerWeights(this);
        }
    }


    private class Backpropagation {
        private final float[] delta;

        private Backpropagation(int cnt) {
            this.delta = new float[cnt];
        }

        float[] calculateLastLayerDelta(float[] expectedOutput) {
            assert inputSum.length == nodeCount() : "Incorrect inputSum count";
            assert delta.length == nodeCount() : "Incorrect delta count";

            for (int j = 0, cnt = nodeCount(); j < cnt; j++) {  // current layer
                float error = expectedOutput[j] - output[j];
                float nodeInputSum = inputSum[j];
                float activationFuncDerivative = func.derivative().apply(nodeInputSum);
                delta[j] = -error * activationFuncDerivative;
            }
            return delta;
        }

        float[] calculateIntermediateLayerDelta(ResponseLayer nextLayer) {
            assert inputSum.length == nodeCount() : "Incorrect inputSum count";
            assert delta.length == nodeCount() : "Incorrect delta count";
            assert nextLayer.backpropagation.delta.length == nextLayer.nodeCount() : "Incorrect delta count";

            for (int j = 0, cnt = nodeCount(); j < cnt; j++) {  // current layer
                float nodeInputSum = inputSum[j];
                float activationFuncDerivative = func.derivative().apply(nodeInputSum);
                float nextLayerDeltaAndWeights = 0;
                assert nextLayer.weight.length == nextLayer.nodeCount() : "Incorrect next layer node count";
                for (int k = 0, nextCnt = nextLayer.nodeCount(); k < nextCnt; k++) {  // next layer
                    assert nextLayer.weight[k].length == nodeCount() : "Incorrect next layer node count";
                    nextLayerDeltaAndWeights += nextLayer.backpropagation.delta[k] * nextLayer.weight[k][j];
                }
                delta[j] = activationFuncDerivative * nextLayerDeltaAndWeights;
            }
            return delta;
        }
    }
}
