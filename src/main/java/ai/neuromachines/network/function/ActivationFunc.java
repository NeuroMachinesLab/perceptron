package ai.neuromachines.network.function;

import java.util.function.Function;

public interface ActivationFunc {

    Function<Float, Float> function();

    Function<Float, Float> derivative();


    static EluFunc elu(float alpha) {
        return EluFunc.of(alpha);
    }

    static IdentityFunc identity(float alpha) {
        return IdentityFunc.of(alpha);
    }

    static PreLuFunc preLu(float alpha) {
        return PreLuFunc.of(alpha);
    }

    static ReLuFunc reLu(float alpha) {
        return ReLuFunc.of(alpha);
    }

    static SigmoidFunc sigmoid(float alpha) {
        return SigmoidFunc.of(alpha);
    }

    static SiLuFunc siLu(float alpha) {
        return SiLuFunc.of(alpha);
    }

    static SoftplusFunc softplus() {
        return SoftplusFunc.of();
    }

    static TanhFunc tanh() {
        return TanhFunc.of();
    }
}
