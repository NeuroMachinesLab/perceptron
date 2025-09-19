package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Softplus
 */
@Getter
@Accessors(fluent = true)
public class SoftplusFunc implements ActivationFunc {

    private final static SoftplusFunc FUNC = new SoftplusFunc();
    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;

    public static SoftplusFunc of() {
        return FUNC;
    }

    private SoftplusFunc() {
        this.function = x -> (float) Math.log(1 + Math.exp(x));
        this.derivative = ActivationFunc.sigmoid(1).function();
    }

    @Override
    public String toString() {
        return "Softplus";
    }
}
