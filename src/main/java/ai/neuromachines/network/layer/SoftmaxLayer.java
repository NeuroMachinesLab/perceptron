package ai.neuromachines.network.layer;

import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.function.SoftmaxFunc;
import lombok.RequiredArgsConstructor;

import static ai.neuromachines.network.function.ActivationFunc.softmax;
import static lombok.AccessLevel.PRIVATE;

/**
 * Implements layer with {@link SoftmaxFunc} activation function
 */
@RequiredArgsConstructor(access = PRIVATE)
class SoftmaxLayer implements ResponseLayer {
    private final ResponseLayerImpl wrapper;

    SoftmaxLayer(float[][] weights, Layer previous) {
        this.wrapper = new ResponseLayerImpl(weights, previous, softmax());
    }

    @Override
    public Layer copyOf(Layer previous) {
        ResponseLayerImpl wrapper = new ResponseLayerImpl(weights().clone(), previous, softmax());
        return new SoftmaxLayer(wrapper);
    }

    @Override
    public ActivationFunc activationFunc() {
        return wrapper.activationFunc();
    }

    @Override
    public float[][] weights() {
        return wrapper.weights();
    }

    @Override
    public int nodeCount() {
        return wrapper.nodeCount();
    }

    @Override
    public float[] input() {
        return wrapper.input();
    }

    @Override
    public float[] output() {
        return wrapper.output();
    }

    @Override
    public void propagate() {
        wrapper.updateInput();
        float[] input = input();
        float[] numerator = new float[input.length];
        float denumerator = 0;
        for (int i = 0; i < input.length; i++) {
            numerator[i] = (float) Math.exp(input[i]);
            denumerator += numerator[i];
        }
        float[] output = output();
        assert input.length == output.length : "Incorrect wrapper layer";
        for (int i = 0; i < output.length; i++) {
            output[i] = numerator[i] / denumerator;
        }
    }
}
