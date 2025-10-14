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
    private final float beta;

    static LeakyReLuFunc of(float alpha, float beta) {
        return new LeakyReLuFunc(alpha, beta);
    }

    private LeakyReLuFunc(float alpha, float beta) {
        this.function = x -> Float.compare(x, 0.0f) < 0 ? beta * x : alpha * x;
        this.derivative = x -> Float.compare(x, 0.0f) < 0 ? beta : alpha;
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public String toString() {
        return "LeakyReLU(alpha=" + alpha + ",beta=" + beta + ")";
    }
}
