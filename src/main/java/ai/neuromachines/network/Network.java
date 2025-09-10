package ai.neuromachines.network;

import ai.neuromachines.Assert;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.layer.IntermediateLayer;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.ResponseLayer;
import ai.neuromachines.network.layer.SensorLayer;
import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.List;

import static lombok.AccessLevel.PRIVATE;

@RequiredArgsConstructor(access = PRIVATE)
public class Network {
    private final List<Layer> layers;

    /**
     * Creates network with many node layers. Each node activation function is set to {@code func} argument.
     * Nodes edges weights are set to random values.
     */
    public static Network of(ActivationFunc func, int... layersNodeCount) {
        Assert.isTrue(layersNodeCount.length > 1, "Minimum 2 layers expected");
        List<Layer> layers = new ArrayList<>();
        SensorLayer sensorLayer = SensorLayer.of(layersNodeCount[0]);
        layers.add(sensorLayer);
        for (int i = 1; i < layersNodeCount.length; i++) {
            int nodeCnt = layersNodeCount[i];
            ResponseLayer layer = ResponseLayer.of(nodeCnt, layers.getLast(), func);
            layers.add(layer);
        }
        return new Network(layers);
    }

    /**
     * @return current input signal
     */
    public float[] input() {
        return sensorLayer().output();
    }

    /**
     * Updates input signal
     */
    public void input(float[] signal) {
        sensorLayer().setOutput(signal);
    }

    /**
     * @return current network output signal
     */
    public float[] output() {
        return outputLayer().output();
    }

    /**
     * @return current i-th layer's output signal
     */
    public float[] output(int layerIndex) {
        Assert.isTrue(layerIndex < layers.size(), "Incorrect layer index");
        return layers.get(layerIndex).output();
    }

    /**
     * @return current layerIndex-th layer's weights
     * @throws IllegalArgumentException if the index corresponds to
     *                                  not the intermediate layer {@code index == 0 || index >= size()})
     */
    public float[][] weights(int layerIndex) {
        Assert.isTrue(layerIndex != 0 && layerIndex < layers.size(), "Incorrect layer index");
        IntermediateLayer layer = (IntermediateLayer) layers.get(layerIndex);
        return layer.weights();
    }

    /**
     * Trains neural network by correcting nodes edges weights
     */
    public void train(float[] expectedOutput) {
        outputLayer().correctWeights(expectedOutput);
    }

    private SensorLayer sensorLayer() {
        return (SensorLayer) layers.getFirst();
    }

    private ResponseLayer outputLayer() {
        return (ResponseLayer) layers.getLast();
    }
}
