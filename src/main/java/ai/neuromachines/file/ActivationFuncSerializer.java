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

    static ActivationFunc deserialize(String s) {
        String funcName = s.split("\\(", 2)[0];
        return switch (funcName.toLowerCase()) {
            case "arctan" -> arctan();
            case "bentidentity" -> bentIdentity();
            case "elu" -> elu(parseAlpha(s));
            case "heaviside" -> heaviside(parseAlpha(s));
            case "gaussian" -> gaussian(parseAlpha(s));
            case "identity" -> identity(parseAlpha(s));
            case "isrlu" -> isrlu(parseAlpha(s));
            case "isru" -> isru(parseAlpha(s));
            case "leakyrelu" -> leakyReLu(parseAlpha(s), parseBeta(s));
            case "relu" -> reLu(parseAlpha(s));
            case "sigmoid" -> sigmoid(parseAlpha(s));
            case "silu" -> siLu(parseAlpha(s));
            case "sinc" -> sinc(parseAlpha(s));
            case "sin" -> sin(parseAlpha(s));
            case "softmax" -> softmax();
            case "softplus" -> softplus();
            case "softsign" -> softsign(parseAlpha(s));
            case "tanh" -> tanh();
            default -> throw new IllegalArgumentException("Activation function is not implemented: " + s);
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
