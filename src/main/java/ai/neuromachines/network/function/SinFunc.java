package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
class SinFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static SinFunc of(float alpha) {
        return new SinFunc(alpha);
    }

    private SinFunc(float alpha) {
        this.function = x -> (float) Math.sin(alpha * x);
        this.derivative = x -> (float) (alpha * Math.cos(alpha * x));
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "Sin(alpha=" + alpha + ")";
    }
}
