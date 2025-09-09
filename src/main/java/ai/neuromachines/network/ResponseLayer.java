package ai.neuromachines.network;

import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.function.ActivationFunc;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
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
     *
     * @param nodeCnt  current layer node count
     * @param previous previous layer
     */
    public ResponseLayer(int nodeCnt, Layer previous, ActivationFunc func) {
        this.previous = previous;
        this.weight = new float[nodeCnt][previous.nodeCount()];
        random(weight);
        this.func = func;
        this.output = new float[nodeCnt];
        this.backpropagation = new Backpropagation(nodeCnt);
    }

    private static void random(float[][] a) {
        for (float[] row : a) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (float) Math.random();
            }
        }
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
        float[] prevOut = previous.output();
        for (int j = 0, cnt = nodeCount(); j < cnt; j++) {  // current layer
            for (int i = 0; i < prevOut.length; i++) {  // previous layer
                float weightDelta = -Constants.trainingVelocity * prevOut[i] * delta[j];
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
            for (int j = 0, cnt = nodeCount(); j < cnt; j++) {  // current layer
                float error = expectedOutput[j] - output[j];
                float nodeInputSum = inputSum[j];
                float activationFuncDerivative = func.derivative().apply(nodeInputSum);
                delta[j] = -error * activationFuncDerivative;
            }
            return delta;
        }

        float[] calculateIntermediateLayerDelta(ResponseLayer nextLayer) {
            for (int j = 0, cnt = nodeCount(); j < cnt; j++) {  // current layer
                float nodeInputSum = inputSum[j];
                float activationFuncDerivative = func.derivative().apply(nodeInputSum);
                float nextLayerDeltaAndWeights = 0;
                for (int k = 0, nextCnt = nextLayer.nodeCount(); k < nextCnt; k++) {  // next layer
                    nextLayerDeltaAndWeights += nextLayer.backpropagation.delta[k] * nextLayer.weight[k][j];
                }
                delta[j] = activationFuncDerivative * nextLayerDeltaAndWeights;
            }
            return delta;
        }
    }
}
