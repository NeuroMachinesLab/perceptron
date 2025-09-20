package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Parametric Rectified Linear Unit and Leaky Rectified Linear Unit (for {@code alpha < 1})
 */
@Getter
@Accessors(fluent = true)
class PreLuFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static PreLuFunc of(float alpha) {
        return new PreLuFunc(alpha);
    }

    private PreLuFunc(float alpha) {
        this.function = x -> Math.max(alpha * x, x);
        this.derivative = x -> Float.compare(x, 0.0f) < 0 ? alpha : 1.0f;
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "PReLU(alpha=" + alpha + ")";
    }
}
