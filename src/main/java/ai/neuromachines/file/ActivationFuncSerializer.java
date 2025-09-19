package ai.neuromachines.file;

import ai.neuromachines.network.function.ActivationFunc;

import static ai.neuromachines.network.function.ActivationFunc.*;

public class ActivationFuncSerializer {

    static String serialize(ActivationFunc func) {
        return func.toString();
    }

    static ActivationFunc deserialize(String string) {
        String funcName = string.split("\\(", 2)[0];
        return switch (funcName.toLowerCase()) {
            case "elu" -> elu(parseAlpha(string));
            case "identity" -> identity(parseAlpha(string));
            case "prelu" -> preLu(parseAlpha(string));
            case "relu" -> reLu(parseAlpha(string));
            case "sigmoid" -> sigmoid(parseAlpha(string));
            case "silu" -> siLu(parseAlpha(string));
            case "softplus" -> softplus();
            case "tanh" -> tanh();
            default -> throw new IllegalArgumentException("Activation function is not implemented: " + string);
        };
    }

    private static float parseAlpha(String string) {
        String arg = string.split("[(,)]")[1];
        String alpha = arg.split("=")[1];
        return Float.parseFloat(alpha);
    }
}
