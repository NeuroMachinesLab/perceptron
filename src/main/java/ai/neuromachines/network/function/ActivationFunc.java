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

    static ActivationFunc leakyReLu() {
        return LeakyReLuFunc.of(0.01f);
    }

    /**
     * @param alpha not learnable coefficient for X < 0; use 0.01 for well known Leaky ReLU
     */
    static ActivationFunc leakyReLu(float alpha) {
        return LeakyReLuFunc.of(alpha);
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

    static ActivationFunc softmax() {
        return SoftmaxFunc.of();
    }

    static ActivationFunc softplus() {
        return SoftplusFunc.of();
    }

    static ActivationFunc tanh() {
        return TanhFunc.of();
    }
}
