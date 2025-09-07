package ai.neuromachines.network.function;

import java.util.function.Function;

public interface ActivationFunc {

    Function<Float, Float> function();

    Function<Float, Float> derivative();

    static SigmoidFunc sigmoid(float alpha) {
        return SigmoidFunc.of(alpha);
    }
}
