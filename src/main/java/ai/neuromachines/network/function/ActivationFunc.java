package ai.neuromachines.network.function;

import java.util.function.Function;

public interface ActivationFunc {

    Function<Float, Float> function();

    Function<Float, Float> derivative();


    static ActivationFunc arctan() {
        return ArctanFunc.of();
    }

    static ActivationFunc bentIdentity() {
        return BentIdentityFunc.of();
    }

    /**
     * @param alpha parameter for X < 0
     */
    static ActivationFunc elu(float alpha) {
        return EluFunc.of(alpha);
    }

    /**
     * @param alpha parameter for X >= 0; use 1 for well known Heaviside Step function
     */
    static ActivationFunc heaviside(float alpha) {
        return HeavisideFunc.of(alpha);
    }

    /**
     * @param alpha parameter for X; use 1 for well known Gaussian function
     */
    static ActivationFunc gaussian(float alpha) {
        return GaussianFunc.of(alpha);
    }

    /**
     * @param alpha parameter for X; use 1 for well known Identity
     */
    static ActivationFunc identity(float alpha) {
        return IdentityFunc.of(alpha);
    }

    /**
     * @param alpha parameter for X in denominator, when X <= 0
     */
    static ActivationFunc isrlu(float alpha) {
        return IsrluFunc.of(alpha);
    }

    /**
     * @param alpha parameter for X in denominator
     */
    static ActivationFunc isru(float alpha) {
        return IsruFunc.of(alpha);
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

    /**
     * @param alpha parameter for X
     */
    static ActivationFunc sinc(float alpha) {
        return SincFunc.of(alpha);
    }

    /**
     * @param alpha parameter for X
     */
    static ActivationFunc sin(float alpha) {
        return SinFunc.of(alpha);
    }

    static ActivationFunc softmax() {
        return SoftmaxFunc.of();
    }

    static ActivationFunc softplus() {
        return SoftplusFunc.of();
    }

    /**
     * @param alpha parameter for X
     */
    static ActivationFunc softsign(float alpha) {
        return SoftsignFunc.of(alpha);
    }

    static ActivationFunc tanh() {
        return TanhFunc.of();
    }
}
