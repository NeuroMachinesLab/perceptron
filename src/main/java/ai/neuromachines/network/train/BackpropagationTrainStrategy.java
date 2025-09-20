package ai.neuromachines.network.train;

import ai.neuromachines.Assert;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.layer.IntermediateLayer;
import ai.neuromachines.network.layer.Layer;
import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.List;

import static lombok.AccessLevel.PRIVATE;

@RequiredArgsConstructor(access = PRIVATE)
class BackpropagationTrainStrategy implements TrainStrategy {
    private final Network network;
    // i-th elements corresponds to network's (i+1) layer deltas
    private final List<LayerDelta> layerDeltas;


    public static BackpropagationTrainStrategy of(Network network) {
        List<LayerDelta> layerDeltas = new ArrayList<>();
        for (IntermediateLayer layer : network.intermediateLayers()) {
            layerDeltas.add(new LayerDelta(layer.nodeCount()));
        }
        return new BackpropagationTrainStrategy(network, layerDeltas);
    }

    @Override
    public void train(float[] expectedOutput) {
        int currentIdx = network.layers().size() - 1;
        IntermediateLayer current = Assert.isInstanceOf(network.layer(currentIdx), IntermediateLayer.class);
        Assert.isTrue(expectedOutput.length == current.nodeCount(), "Incorrect excepted output count");
        Assert.isTrue(current.input() != null, "Call output() first, there is no calculated output signal for weighs correcting");
        float[] updatedDelta = getLayerDelta(currentIdx)
                .updateLastLayerDelta(current.input(), current.output(), expectedOutput, current.activationFunc());
        Layer previous = network.layer(currentIdx - 1);
        updateWeights(previous.output(), updatedDelta, current.weights());
        updatePreviousLayerWeights(currentIdx);
    }

    private LayerDelta getLayerDelta(int layerIdx) {
        return layerDeltas.get(layerIdx - 1);
    }

    private void updatePreviousLayerWeights(int currentLayerIdx) {
        int previousIdx = currentLayerIdx - 1;
        if (previousIdx > 0) {  // don't update weights for sensor layer (idx == 0)
            Layer previous = network.layer(previousIdx);
            if (previous instanceof IntermediateLayer) {
                correctIntermediateLayerWeights(previousIdx);
            }
        }
    }

    private void correctIntermediateLayerWeights(int currentIdx) {
        IntermediateLayer current = Assert.isInstanceOf(network.layer(currentIdx), IntermediateLayer.class);
        IntermediateLayer next = Assert.isInstanceOf(network.layer(currentIdx + 1), IntermediateLayer.class);
        LayerDelta nextLayerDeltas = getLayerDelta(currentIdx + 1);
        float[] updatedDelta = getLayerDelta(currentIdx)
                .updateIntermediateLayerDelta(current.input(), current.activationFunc(), next.weights(), nextLayerDeltas);
        Layer previous = network.layer(currentIdx - 1);
        updateWeights(previous.output(), updatedDelta, current.weights());
        updatePreviousLayerWeights(currentIdx);
    }

    /**
     * @param previousLayerOutput (j-1) layer output
     * @param delta               j-th layer delta
     * @param weight              weights between (j-1) and j-th layers
     *                            (matrix rows count equals to j-th layer node count, cols count equals to (j-1) layer node count)
     */
    private static void updateWeights(float[] previousLayerOutput, float[] delta, float[][] weight) {
        assert delta.length == weight.length : "Incorrect delta count";

        for (int j = 0; j < weight.length; j++) {  // current layer
            assert weight[j].length == previousLayerOutput.length : "Incorrect previous layer node count";
            for (int i = 0; i < previousLayerOutput.length; i++) {  // previous layer
                float weightDelta = -Constants.TRAINING_VELOCITY * previousLayerOutput[i] * delta[j];
                weight[j][i] += weightDelta;
            }
        }
    }


    public static class LayerDelta {
        private final float[] delta;

        private LayerDelta(int cnt) {
            this.delta = new float[cnt];
        }

        /**
         * @param input          j-th layer's node input signal
         * @param output         j-th layer's node output signal
         * @param expectedOutput j-th layer's node expected output signal
         * @param func           j-th layer's node activation function
         * @return j-th layer's node updates deltas
         */
        float[] updateLastLayerDelta(float[] input, float[] output, float[] expectedOutput, ActivationFunc func) {
            assert delta.length == input.length : "Incorrect delta count";

            for (int j = 0, cnt = input.length; j < cnt; j++) {  // current layer
                float error = expectedOutput[j] - output[j];
                float nodeInputSum = input[j];
                float activationFuncDerivative = func.derivative().apply(nodeInputSum);
                delta[j] = -error * activationFuncDerivative;
            }
            return delta;
        }

        /**
         * @param input            j-th layer nodes input signal
         * @param func             j-th layer's node activation function
         * @param nextLayerWeights weights between j-th and (j+1) layer
         *                         (matrix rows count equals to (j+1) layer node count, cols count equals to j-th layer node count)
         * @param nextLayerDeltas  (j+1) layer deltas
         * @return j-th layer's node updates deltas
         */
        float[] updateIntermediateLayerDelta(float[] input, ActivationFunc func, float[][] nextLayerWeights, LayerDelta nextLayerDeltas) {
            assert delta.length == input.length : "Incorrect delta count";
            assert nextLayerDeltas.delta.length == nextLayerWeights.length : "Incorrect next layer delta count";

            for (int j = 0; j < input.length; j++) {  // current layer
                float nodeInputSum = input[j];
                float activationFuncDerivative = func.derivative().apply(nodeInputSum);
                float nextLayerDeltaAndWeights = 0;
                for (int k = 0; k < nextLayerWeights.length; k++) {  // next layer
                    assert nextLayerWeights[k].length == input.length : "Incorrect next layer node count";
                    nextLayerDeltaAndWeights += nextLayerDeltas.delta[k] * nextLayerWeights[k][j];
                }
                delta[j] = activationFuncDerivative * nextLayerDeltaAndWeights;
            }
            return delta;
        }
    }
}
