package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Leaky Rectified Linear Unit with {@code alpha} coefficient for negative values
 */
@Getter
@Accessors(fluent = true)
class LeakyReLuFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static LeakyReLuFunc of(float alpha) {
        return new LeakyReLuFunc(alpha);
    }

    private LeakyReLuFunc(float alpha) {
        this.function = x -> Float.compare(x, 0.0f) < 0 ? alpha * x : x;
        this.derivative = x -> Float.compare(x, 0.0f) < 0 ? alpha : 1.0f;
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "LeakyReLU(alpha=" + alpha + ")";
    }
}
