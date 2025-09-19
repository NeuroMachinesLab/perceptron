package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Sigmoid-weighted Linear Unit
 */
@Getter
@Accessors(fluent = true)
public class SiLuFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    public static SiLuFunc of(float alpha) {
        return new SiLuFunc(alpha);
    }

    private SiLuFunc(float alpha) {
        Function<Float, Float> sigmoid = ActivationFunc.sigmoid(alpha).function();
        this.function = x -> x * sigmoid.apply(x);
        this.derivative = x -> {
            float y = sigmoid.apply(x);
            return y + x * alpha * y * (1 - y);
        };
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "SiLU(alpha=" + alpha + ")";
    }
}
