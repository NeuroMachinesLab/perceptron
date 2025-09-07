package ai.neuromachines.network.function;

import java.util.function.Function;

public class SigmoidFunc implements ActivationFunc {

    private final float alpha;
    private final Function<Float, Float> f;


    public static SigmoidFunc of(float alpha) {
        return new SigmoidFunc(alpha);
    }

    private SigmoidFunc(float alpha) {
        this.alpha = alpha;
        this.f = x -> (float) (1 / (1 + Math.exp(-2 * alpha * x)));
    }

    @Override
    public Function<Float, Float> function() {
        return f;
    }

    @Override
    public Function<Float, Float> derivative() {
        return x -> {
            float y = f.apply(x);
            return 2 * alpha * y * (1 - y);
        };
    }
}
