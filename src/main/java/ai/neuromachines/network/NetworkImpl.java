package ai.neuromachines.network;

import ai.neuromachines.Assert;
import ai.neuromachines.network.layer.IntermediateLayer;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.ResponseLayer;
import ai.neuromachines.network.layer.SensorLayer;
import lombok.RequiredArgsConstructor;

import java.util.List;

import static lombok.AccessLevel.PACKAGE;

@RequiredArgsConstructor(access = PACKAGE)
public class NetworkImpl implements Network {
    private final List<Layer> layers;

    @Override
    public int layersCount() {
        return layers.size();
    }

    @Override
    public float[] input() {
        return sensorLayer().output();
    }

    @Override
    public void input(float[] signal) {
        sensorLayer().setOutput(signal);
    }

    @Override
    public float[] output() {
        return outputLayer().output();
    }


    @Override
    public float[] output(int layerIndex) {
        Assert.isTrue(layerIndex < layers.size(), "Incorrect layer index");
        return layers.get(layerIndex).output();
    }

    @Override
    public float[][] weights(int layerIndex) {
        Assert.isTrue(layerIndex > 0 && layerIndex < layers.size(), "Incorrect layer index");
        IntermediateLayer layer = (IntermediateLayer) layers.get(layerIndex);
        return layer.weights();
    }

    @Override
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
