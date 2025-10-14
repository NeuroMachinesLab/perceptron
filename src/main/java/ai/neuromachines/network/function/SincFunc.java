package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
class SincFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static SincFunc of(float alpha) {
        return new SincFunc(alpha);
    }

    private SincFunc(float alpha) {
        this.function = x -> {
            if (x == 0) {
                return 1f;
            }
            float t = alpha * x;
            return (float) (Math.sin(t) / t);
        };
        this.derivative = x -> {
            if (x == 0) {
                return 0f;
            }
            float t = alpha * x;
            return (float) (Math.cos(t) - Math.sin(t) / t) / t;
        };
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "Sinc(alpha=" + alpha + ")";
    }
}
