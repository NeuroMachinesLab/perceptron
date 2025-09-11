package ai.neuromachines.network;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.ResponseLayer;
import ai.neuromachines.network.layer.SensorLayer;

import java.util.ArrayList;
import java.util.List;

public interface Network {

    /**
     * Creates network with {@code layersNodeCount.length} node layers.
     * Each node activation function is set to {@code func} argument.
     * Nodes edges weights are set to random values.
     */
    static Network of(ActivationFunc func, int... layersNodeCount) {
        Assert.isTrue(layersNodeCount.length > 1, "Minimum 2 layers expected");
        List<Layer> layers = new ArrayList<>();
        SensorLayer sensorLayer = SensorLayer.of(layersNodeCount[0]);
        layers.add(sensorLayer);
        for (int i = 1; i < layersNodeCount.length; i++) {
            int nodeCnt = layersNodeCount[i];
            ResponseLayer layer = ResponseLayer.of(nodeCnt, layers.getLast(), func);
            layers.add(layer);
        }
        return new NetworkImpl(layers);
    }

    /**
     * Creates network with {@code weights.length + 1} node layers.
     * Each node activation function is set to {@code func} argument.
     * Nodes edges weights are initialized by {@code weights} argument.
     *
     * <p>{@code weights} 0-th element corresponds to weights matrix between 0-th and 1-th node layers.
     * This matrix row count is equals to node count in 0-th layer, column count is equals to node count in 1-th layer.
     *
     * <p>{@code weights} 1-th element corresponds to weights matrix between 1-th and 2-th node layers and so on.
     */
    static Network of(ActivationFunc func, float[][]... weights) {
        Assert.isTrue(weights.length > 0, "Minimum 2 layers expected");
        List<Layer> layers = new ArrayList<>();
        int sensorLayerNodeCount = weights[0].length;
        SensorLayer sensorLayer = SensorLayer.of(sensorLayerNodeCount);
        layers.add(sensorLayer);
        for (float[][] weight :weights) {
            float[][] transposedWeight = Matrix.transpose(weight);
            ResponseLayer layer = ResponseLayer.of(transposedWeight, layers.getLast(), func);
            layers.add(layer);
        }
        return new NetworkImpl(layers);
    }

    int layersCount();

    /**
     * @return sensor layer signals
     */
    float[] input();

    /**
     * Sets sensor layer signals
     */
    void input(float[] signal);

    /**
     * @return output layer signals
     */
    float[] output();

    /**
     * @return output n-th (sensor for 0, hidden or output for last) layer signals
     * @throws IllegalArgumentException if {@code layerIndex} is < 0 or >= {@link #layersCount()}
     */
    float[] output(int layerIndex);

    /**
     * @return weights for connections between i-th and (i-1) layer nodes
     * (row count equals to i-th layer node count, cols count equals to (i-1) layer node count)
     * @throws IllegalArgumentException if the {@code layerIndex} doesn't correspond to
     *                                  the intermediate layer {@code layerIndex <= 0 || layerIndex >=} {@link #layersCount()}
     */
    float[][] weights(int layerIndex);

    /**
     * Trans network for expected output
     */
    void train(float[] expectedOutput);
}
