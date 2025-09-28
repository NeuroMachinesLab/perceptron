package ai.neuromachines.network.train.lossfunc;

import ai.neuromachines.Assert;
import ai.neuromachines.network.function.ActivationFunc;

/**
 * Allows to minimize errors for rare events
 *
 * @see <a href="https://en.wikipedia.org/wiki/Cross-entropy">Definition</a>
 * @see <a href="https://habr.com/ru/articles/155235">Evaluation</a>
 */
class CrossEntropyLossFunc implements LossFunc {
    static final CrossEntropyLossFunc INSTANCE = new CrossEntropyLossFunc();

    static CrossEntropyLossFunc of() {
        return INSTANCE;
    }

    @Override
    public void verifyExpectedOutput(float[] expectedOutput) {
        float total = 0;
        for (float v : expectedOutput) {
            total += v;
        }
        Assert.isTrue(0.99f < total && total < 1.01f,
                "For 'softmax' activation function in output layer sum of expected output should equals to 1.00");
    }

    @Override
    public float calculateDelta(float input, float output, float expectedOutput, ActivationFunc func) {
        return output - expectedOutput;
    }
}
