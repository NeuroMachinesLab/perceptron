package ai.neuromachines.network.function;

import java.util.function.Function;

public class SigmoidFunc implements ActivationFunc {

    private final Function<Float, Float> f;
    private final Function<Float, Float> derivative;


    public static SigmoidFunc of(float alpha) {
        return new SigmoidFunc(alpha);
    }

    private SigmoidFunc(float alpha) {
        double doubleAlpha = 2 * alpha;
        this.f = x -> (float) (1 / (1 + Math.exp(-doubleAlpha * x)));
        this.derivative = x -> {
            float y = f.apply(x);
            return (float) (doubleAlpha * y * (1 - y));
        };

    }

    @Override
    public Function<Float, Float> function() {
        return f;
    }

    @Override
    public Function<Float, Float> derivative() {
        return derivative;
    }
}
