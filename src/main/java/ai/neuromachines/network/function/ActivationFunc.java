package ai.neuromachines.network.function;

import java.util.function.Function;

public interface ActivationFunc {

    Function<Float, Float> function();

    Function<Float, Float> derivative();


    /**
     * @param alpha parameter for X < 0
     */
    static ActivationFunc elu(float alpha) {
        return EluFunc.of(alpha);
    }

    /**
     * @param alpha use 1 for well known Identity
     */
    static ActivationFunc identity(float alpha) {
        return IdentityFunc.of(alpha);
    }

    static ActivationFunc leakyReLu() {
        return LeakyReLuFunc.of(1.0f, 0.01f);
    }

    /**
     * @param alpha parameter for X >= 0; use 1 for well known Leaky ReLU
     * @param beta parameter for X < 0; use 0.01 for well known Leaky ReLU
     */
    static ActivationFunc leakyReLu(float alpha, float beta) {
        return LeakyReLuFunc.of(alpha, beta);
    }

    /**
     * @param alpha parameter for X >= 0; use 1 for well known ReLU
     */
    static ActivationFunc reLu(float alpha) {
        return ReLuFunc.of(alpha);
    }

    /**
     * @param alpha parameter for X; use 1 for well known Sigmoid
     */
    static ActivationFunc sigmoid(float alpha) {
        return SigmoidFunc.of(alpha);
    }

    /**
     * @param alpha parameter for exponent argument; use 1 for well known SiLU
     */
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
