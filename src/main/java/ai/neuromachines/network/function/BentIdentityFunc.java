package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
class BentIdentityFunc implements ActivationFunc {

    private final static BentIdentityFunc FUNC = new BentIdentityFunc();
    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;

    static BentIdentityFunc of() {
        return FUNC;
    }

    private BentIdentityFunc() {
        this.function = x -> (float) ((Math.sqrt(x * x + 1) - 1) / 2 + x);
        this.derivative = x -> (float) (x / (2 * Math.sqrt(x * x + 1)) + 1);
    }

    @Override
    public String toString() {
        return "BentIdentity";
    }
}
