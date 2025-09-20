package ai.neuromachines.network.layer;

import ai.neuromachines.network.function.ActivationFunc;

/**
 * Intermediate and Output layers representation.
 * This layer in comparison with {@link SensorLayer} has input values, weights and activation function.
 */
public interface ResponseLayer extends Layer {

    /**
     * Creates layer with specified connection weights.
     *
     * @param weights  weights for connections with nodes of the previous layer
     *                 (row count equals to current layer node count, cols count equals to previous layer node count)
     * @param previous previous layer
     * @param func     activation function
     */
    static ResponseLayer of(float[][] weights, Layer previous, ActivationFunc func) {
        return new ResponseLayerImpl(weights, previous, func);
    }

    /**
     * Creates layer with random connection weights.
     *
     * @param nodeCnt  current layer node count
     * @param previous previous layer
     * @param func     activation function
     */
    static ResponseLayer of(int nodeCnt, Layer previous, ActivationFunc func) {
        float[][] weights = randomWeights(nodeCnt, previous.nodeCount());
        return new ResponseLayerImpl(weights, previous, func);
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

    ActivationFunc activationFunc();

    /**
     * Returns weights for connections between this layer and previous layer nodes
     * (row count equals to this layer node count, cols count equals to previous layer node count)
     */
    float[][] weights();
}
