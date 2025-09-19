package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
public class SigmoidFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    public static SigmoidFunc of(float alpha) {
        return new SigmoidFunc(alpha);
    }

    private SigmoidFunc(float alpha) {
        this.function = x -> (float) (1 / (1 + Math.exp(-alpha * x)));
        this.derivative = x -> {
            float y = function.apply(x);
            return alpha * y * (1 - y);
        };
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "Sigmoid(alpha=" + alpha + ")";
    }
}
