package ai.neuromachines.network;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.layer.IntermediateLayer;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.SensorLayer;

import java.util.List;
import java.util.SequencedCollection;

public class NetworkImpl implements Network {
    private final List<Layer> layers;

    NetworkImpl(SequencedCollection<Layer> layers) {
        Assert.isTrue(layers.size() > 1, "At least 2 layers expected");
        boolean isOtherLayerIsIntermediate = layers.stream()
                .skip(1)  // first layer may be SensorLayer or IntermediateLayer
                .allMatch(layer -> layer instanceof IntermediateLayer);
        Assert.isTrue(isOtherLayerIsIntermediate, "Second and next layers should be IntermediateLayer");
        this.layers = List.copyOf(layers);
    }

    @Override
    public List<Layer> layers() {
        return layers;
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<IntermediateLayer> intermediateLayers() {
        Object intermediateLayers = layers.subList(1, layers().size());
        return (List<IntermediateLayer>) intermediateLayers;  // checked in constructor
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
        float[][] w = transposedWeights(layerIndex + 1);
        return Matrix.transpose(w);
    }

    @Override
    public float[][] transposedWeights(int layerIndex) {
        Assert.isTrue(layerIndex < layers.size(), "Incorrect layer index");
        Layer layer = layers.get(layerIndex);
        Assert.isTrue(layer instanceof IntermediateLayer il, "Layer at given index is not IntermediateLayer");
        return ((IntermediateLayer) layer).weights();

    }

    @Override
    public void train(float[] expectedOutput) {
        outputLayer().train(expectedOutput);
    }

    private SensorLayer sensorLayer() {
        Layer first = layers.getFirst();
        Assert.isTrue(first instanceof SensorLayer, "First layer is not SensorLayer");
        return (SensorLayer) first;
    }

    private IntermediateLayer outputLayer() {
        return (IntermediateLayer) layers.getLast();
    }
}
