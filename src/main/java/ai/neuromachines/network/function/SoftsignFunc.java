package ai.neuromachines.network.function;

import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Getter
@Accessors(fluent = true)
class SoftsignFunc implements ActivationFunc {

    private final Function<Float, Float> function;
    private final Function<Float, Float> derivative;
    private final float alpha;

    static SoftsignFunc of(float alpha) {
        return new SoftsignFunc(alpha);
    }

    private SoftsignFunc(float alpha) {
        this.function = x -> {
            float t = alpha * x;
            return t / (1 + Math.abs(t));
        };
        this.derivative = x -> (float) (alpha / Math.pow(1 + Math.abs(alpha * x), 2));
        this.alpha = alpha;
    }

    @Override
    public String toString() {
        return "Softsign(alpha=" + alpha + ")";
    }
}
