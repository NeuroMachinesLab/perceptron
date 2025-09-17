package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Hyperbolic tangent
 */
@Getter
@Accessors(fluent = true)
public class TanhFunc implements ActivationFunc {

    private final static TanhFunc FUNC = new TanhFunc();
    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;

    public static TanhFunc of() {
        return FUNC;
    }

    private TanhFunc() {
        this.function = x -> {
            double ex = Math.exp(x);
            double e_x = Math.exp(-x);
            return (float) ((ex - e_x) / (ex + e_x));
        };
        this.derivative = x -> {
            float y = function.apply(x);
            return 1 - y * y;
        };
    }
}
