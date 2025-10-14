package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Inverse square root linear unit
 */
@Getter
@Accessors(fluent = true)
class IsrluFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static IsrluFunc of(float alpha) {
        return new IsrluFunc(alpha);
    }

    private IsrluFunc(float alpha) {
        ActivationFunc isru = ActivationFunc.isru(alpha);
        this.function = x -> Float.compare(x, 0.0f) < 0 ? isru.function().apply(x) : x;
        this.derivative = x -> Float.compare(x, 0.0f) < 0 ? isru.derivative().apply(x) : 1;
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "ISRLU(alpha=" + alpha + ")";
    }
}
