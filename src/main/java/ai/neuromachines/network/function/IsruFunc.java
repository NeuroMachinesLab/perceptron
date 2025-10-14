package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Inverse square root unit
 */
@Getter
@Accessors(fluent = true)
class IsruFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static IsruFunc of(float alpha) {
        return new IsruFunc(alpha);
    }

    private IsruFunc(float alpha) {
        this.function = x -> (float) (x / Math.sqrt(1 + alpha * x * x));
        this.derivative = x -> (float) (1 / Math.pow(1 + alpha * x * x, 1.5));
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "ISRU(alpha=" + alpha + ")";
    }
}
