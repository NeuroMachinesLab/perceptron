package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
public class IdentityFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    public static IdentityFunc of(float alpha) {
        return new IdentityFunc(alpha);
    }

    private IdentityFunc(float alpha) {
        this.function = x -> alpha * x;
        this.derivative = _ -> alpha;
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "Identity(alpha=" + alpha + ")";
    }
}
