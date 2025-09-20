package ai.neuromachines.network;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.layer.IntermediateLayer;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.SensorLayer;

import java.util.List;
import java.util.SequencedCollection;

class NetworkImpl implements Network {
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
    public Layer layer(int i) {
        Assert.isTrue(i >= 0 && i < layers.size(), "Incorrect layer index");
        return layers.get(i);
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
        return layers.getFirst().input();
    }

    @Override
    public void input(float[] signal) {
        Assert.isInstanceOf(layers.getFirst(), SensorLayer.class)
                .setInput(signal);
    }

    @Override
    public float[] input(int layerIndex) {
        return layer(layerIndex).input();
    }

    @Override
    public float[] output() {
        return layers.getLast().output();
    }

    @Override
    public float[] output(int layerIndex) {
        return layer(layerIndex).output();
    }

    @Override
    public float[][] weights(int layerIndex) {
        float[][] w = transposedWeights(layerIndex + 1);
        return Matrix.transpose(w);
    }

    private float[][] transposedWeights(int layerIndex) {
        return Assert.isInstanceOf(layer(layerIndex), IntermediateLayer.class)
                .weights();

    }
}
