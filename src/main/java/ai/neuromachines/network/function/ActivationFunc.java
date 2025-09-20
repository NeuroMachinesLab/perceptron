package ai.neuromachines.network.function;

import java.util.function.Function;

public interface ActivationFunc {

    Function<Float, Float> function();

    Function<Float, Float> derivative();


    static ActivationFunc elu(float alpha) {
        return EluFunc.of(alpha);
    }

    static ActivationFunc identity(float alpha) {
        return IdentityFunc.of(alpha);
    }

    static ActivationFunc preLu(float alpha) {
        return PreLuFunc.of(alpha);
    }

    static ActivationFunc reLu(float alpha) {
        return ReLuFunc.of(alpha);
    }

    static ActivationFunc sigmoid(float alpha) {
        return SigmoidFunc.of(alpha);
    }

    static ActivationFunc siLu(float alpha) {
        return SiLuFunc.of(alpha);
    }

    static ActivationFunc softplus() {
        return SoftplusFunc.of();
    }

    static ActivationFunc tanh() {
        return TanhFunc.of();
    }
}
