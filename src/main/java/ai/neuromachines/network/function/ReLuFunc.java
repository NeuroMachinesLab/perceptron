package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Rectified Linear Unit
 */
@Getter
@Accessors(fluent = true)
class ReLuFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static ReLuFunc of(float alpha) {
        return new ReLuFunc(alpha);
    }

    private ReLuFunc(float alpha) {
        this.function = x -> Math.max(0, alpha * x);
        this.derivative = x -> Float.compare(x, 0.0f) < 0 ? 0.0f : alpha;
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "ReLU(alpha=" + alpha + ")";
    }
}
