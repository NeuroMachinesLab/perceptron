package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Hyperbolic tangent
 */
@Getter
@Accessors(fluent = true)
class ArctanFunc implements ActivationFunc {

    private final static ArctanFunc FUNC = new ArctanFunc();
    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;

    static ArctanFunc of() {
        return FUNC;
    }

    private ArctanFunc() {
        this.function = x -> (float) Math.atan(x);
        this.derivative = x -> 1 / (x * x + 1);
    }

    @Override
    public String toString() {
        return "Arctan";
    }
}
