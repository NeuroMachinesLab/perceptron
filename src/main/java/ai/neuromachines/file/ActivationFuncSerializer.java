package ai.neuromachines.file;

import ai.neuromachines.network.function.ActivationFunc;

import static ai.neuromachines.network.function.ActivationFunc.*;

/**
 * Serializes activation function as string in one of the format
 * <pre>
 *     {@literal <func_name>}
 * </pre>
 * or
 * <pre>
 *     {@literal <func_name>(<arg_name>=<arg_value>)}
 * </pre>
 */
public class ActivationFuncSerializer {

    static String serialize(ActivationFunc func) {
        return func.toString();
    }

    static ActivationFunc deserialize(String string) {
        String funcName = string.split("\\(", 2)[0];
        return switch (funcName.toLowerCase()) {
            case "elu" -> elu(parseAlpha(string));
            case "identity" -> identity(parseAlpha(string));
            case "leakyrelu" -> leakyReLu(parseAlpha(string), parseBeta(string));
            case "relu" -> reLu(parseAlpha(string));
            case "sigmoid" -> sigmoid(parseAlpha(string));
            case "silu" -> siLu(parseAlpha(string));
            case "softmax" -> softmax();
            case "softplus" -> softplus();
            case "tanh" -> tanh();
            default -> throw new IllegalArgumentException("Activation function is not implemented: " + string);
        };
    }

    private static float parseAlpha(String string) {
        return parseArg(string, 1);
    }

    private static float parseBeta(String string) {
        return parseArg(string, 2);
    }

    private static float parseArg(String string, int argNum) {
        String arg = string.split("[(,)]")[argNum];
        String beta = arg.split("=")[1];
        return Float.parseFloat(beta);
    }
}
