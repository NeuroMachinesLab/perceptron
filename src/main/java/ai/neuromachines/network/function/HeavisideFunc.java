package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
class HeavisideFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static HeavisideFunc of(float alpha) {
        return new HeavisideFunc(alpha);
    }

    private HeavisideFunc(float alpha) {
        this.function = x -> Float.compare(x, 0.0f) < 0 ? 0 : alpha;
        this.derivative = _ -> 0.0f;
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "Heaviside(alpha=" + alpha + ")";
    }
}
