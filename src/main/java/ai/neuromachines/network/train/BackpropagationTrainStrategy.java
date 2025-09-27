package ai.neuromachines.network.train;

import ai.neuromachines.Assert;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.function.SoftmaxFunc;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.ResponseLayer;
import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static lombok.AccessLevel.PRIVATE;

/**
 * Squared error loss function is used with one exception.
 * <p>
 * If output layer's activation function is {@link SoftmaxFunc}, then cross-entropy loss function is used.
 * Check out <a href="https://habr.com/ru/articles/155235">this article</a> for backpropagation method calculation
 * in this case.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Backpropagation">Backpropagation</a>
 */
@RequiredArgsConstructor(access = PRIVATE)
class BackpropagationTrainStrategy implements TrainStrategy {
    private final Network network;
    // i-th elements corresponds to network's (i+1) layer deltas
    private final List<LayerDelta> layerDeltas;


    public static BackpropagationTrainStrategy of(Network network) {
        List<LayerDelta> layerDeltas = new ArrayList<>();
        for (ResponseLayer layer : network.responseLayers()) {
            layerDeltas.add(new LayerDelta(layer.nodeCount()));
        }
        return new BackpropagationTrainStrategy(network, layerDeltas);
    }

    @Override
    public void train(float[] input, float[] expectedOutput) {
        network.input(input);
        network.propagate();
        int currentIdx = network.layers().size() - 1;
        ResponseLayer current = Assert.isInstanceOf(network.layer(currentIdx), ResponseLayer.class);
        Assert.isTrue(expectedOutput.length == current.nodeCount(), "Incorrect excepted output count");
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
            correctIntermediateLayerWeights(previousIdx);
        }
    }

    private void correctIntermediateLayerWeights(int currentIdx) {
        ResponseLayer current = Assert.isInstanceOf(network.layer(currentIdx), ResponseLayer.class);
        ResponseLayer next = Assert.isInstanceOf(network.layer(currentIdx + 1), ResponseLayer.class);
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
                float weightDelta = -Constants.LEARNING_RATE * previousLayerOutput[i] * delta[j];
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
                float error = output[j] - expectedOutput[j];
                if (Objects.equals(func, ActivationFunc.softmax())) {
                    isSumEqualsToOne(expectedOutput);
                    delta[j] = error;  // Gross-entropy loss delta
                } else {
                    float nodeInputSum = input[j];
                    float activationFuncDerivative = func.derivative().apply(nodeInputSum);
                    delta[j] = error * activationFuncDerivative;  // Squared error loss delta
                }
            }
            return delta;
        }

        private static void isSumEqualsToOne(float[] expectedOutput) {
            float total = 0;
            for (float v : expectedOutput) {
                total += v;
            }
            Assert.isTrue(0.99f < total && total < 1.01f,
                    "For 'softmax' activation function in output layer sum of expected output should equals to 1.00");
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
