package ai.neuromachines.network.function;

import java.util.function.Function;

/**
 * Stub class.
 * <p>
 * Softmax function requires all nodes' inputs of the layer (requires float[] as an argument).
 * Should be used by special type of layer {@link ai.neuromachines.network.layer.SoftmaxLayer}.
 * Don't pass to {@link ai.neuromachines.network.layer.ResponseLayerImpl} as an argument.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Activation_function">Activation Function</a>
 */
public class SoftmaxFunc implements ActivationFunc {

    private static final SoftmaxFunc FUNC = new SoftmaxFunc();

    static SoftmaxFunc of() {
        return FUNC;
    }

    @Override
    public Function<Float, Float> function() {
        throw new UnsupportedOperationException("Can't by implemented by Function<Float, Float>");
    }

    @Override
    public Function<Float, Float> derivative() {
        throw new UnsupportedOperationException("Can't by implemented by Function<Float, Float>");
    }

    @Override
    public String toString() {
        return "Softmax";
    }
}
