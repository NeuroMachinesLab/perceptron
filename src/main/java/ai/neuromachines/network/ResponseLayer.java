package ai.neuromachines.network;

import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.function.ActivationFunc;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ResponseLayer implements Layer {
    private final Layer previous;
    // 1-st row contains all weights for 1-st node of current layer to the previous layer
    private final float[][] weight;
    private final ActivationFunc func;
    // current output values of all nodes in layer
    private float[] output;

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
        float[] sum = Matrix.multiply(weight, previous.output());
        return output = Matrix.applyFunc(sum, func.function());
    }


}
