package ai.neuromachines.network;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.ResponseLayer;
import ai.neuromachines.network.layer.SensorLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.SequencedCollection;

public interface Network {


    /**
     * Creates network with specified node layers.
     * Each node layer may have own activation function.
     */
    static Network of(SequencedCollection<Layer> layers) {
        return new NetworkImpl(layers);
    }

    /**
     * Creates network with {@code layersNodeCount.length} node layers.
     * All layers (except sensor layer) activation function are set to {@code func} argument.
     * Nodes edges weights are set to random values.
     */
    static Network of(List<ActivationFunc> func, int... layersNodeCount) {
        Assert.isTrue(layersNodeCount.length > 1, "Minimum 2 layers expected");
        Assert.isTrue(func.size() + 1 == layersNodeCount.length, "Incorrect activation function count");
        List<Layer> layers = new ArrayList<>();
        SensorLayer sensorLayer = SensorLayer.of(layersNodeCount[0]);
        layers.add(sensorLayer);
        for (int i = 1; i < layersNodeCount.length; i++) {
            int nodeCnt = layersNodeCount[i];
            ResponseLayer layer = ResponseLayer.of(nodeCnt, layers.getLast(), func.get(i - 1));
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
        for (float[][] weight : weights) {
            float[][] transposedWeight = Matrix.transpose(weight);
            ResponseLayer layer = ResponseLayer.of(transposedWeight, layers.getLast(), func);
            layers.add(layer);
        }
        return new NetworkImpl(layers);
    }

    /**
     * @throws IllegalArgumentException if {@code layerIndex < 0} or {@code layerIndex >= layers count}
     */
    Layer layer(int i);

    List<Layer> layers();

    /**
     * Same layers as for {@link #layers()} but without first one
     */
    List<ResponseLayer> responseLayers();

    /**
     * @return sensor layer input signals
     */
    float[] input();

    /**
     * Sets sensor layer signals
     */
    void input(float[] signal);

    /**
     * @return input for n-th (sensor for 0, hidden or output for last) layer
     * @throws IllegalArgumentException if {@code layerIndex} is < 0 or >= layers count
     */
    float[] input(int layerIndex);

    /**
     * @return output layer signals
     */
    float[] output();

    /**
     * @return output for n-th (sensor for 0, hidden or output for last) layer
     * @throws IllegalArgumentException if {@code layerIndex} is < 0 or >= layers count
     */
    float[] output(int layerIndex);

    /**
     * @return weights for connections between i-th and (i+1) layer nodes
     * (matrix row count equals to i-th layer node count, cols count equals to (i+1) layer node count)
     * @throws IllegalArgumentException if {@code layerIndex < 0} or {@code layerIndex >= (layers count - 1)}
     */
    float[][] weights(int layerIndex);

    /**
     * Propagates input signals from sensor to output layer
     */
    void propagate();
}
