package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

/**
 * Softplus
 */
@Getter
@Accessors(fluent = true)
public class SoftPlusFunc implements ActivationFunc {

    private final static SoftPlusFunc FUNC = new SoftPlusFunc();
    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;

    public static SoftPlusFunc of() {
        return FUNC;
    }

    private SoftPlusFunc() {
        this.function = x -> (float) Math.log(1 + Math.exp(x));
        this.derivative = ActivationFunc.sigmoid(1).function();
    }
}
