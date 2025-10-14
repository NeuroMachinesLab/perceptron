package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
class GaussianFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static GaussianFunc of(float alpha) {
        return new GaussianFunc(alpha);
    }

    private GaussianFunc(float alpha) {
        this.function = x -> (float) Math.exp(-alpha * x * x);
        this.derivative = x -> (float) (-2 * alpha * Math.exp(-alpha * x * x));
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "Gaussian(alpha=" + alpha + ")";
    }
}
