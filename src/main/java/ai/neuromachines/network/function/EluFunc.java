package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Exponential Linear Unit
 */
@Getter
@Accessors(fluent = true)
public class EluFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    public static EluFunc of(float alpha) {
        return new EluFunc(alpha);
    }

    private EluFunc(float alpha) {
        this.function = x -> (float) (Float.compare(x, 0.0f) < 0 ? alpha * (Math.exp(x) - 1) : x);
        this.derivative = x -> (float) (Float.compare(x, 0.0f) < 0 ? alpha * Math.exp(x) : 1.0f);
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "ELU(alpha=" + alpha + ")";
    }
}
